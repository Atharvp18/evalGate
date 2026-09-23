"""Phase 9 regression-injection study.

For each of the 10 injections in `injections.py`, runs the eval suite twice —
once with the real agent (a "before" snapshot, computed once and shared
across all injections) and once with the injected regression applied (the
"after" snapshot) — under two scoring regimes:

  full mode:  each case's own trial count, the full scorer set (contains,
              regex, numeric, trajectory, judge), and the real two-condition
              regression gate (margin AND CI exclusion) from compare.py.
  naive mode: 1 trial per case, deterministic scorers only (contains/regex/
              numeric — no trajectory, no judge, since a team without an eval
              framework is unlikely to have built either), and a naive gate:
              any case that passed before and fails after is flagged. No
              margin, no CI — that absence is exactly what naive mode is a
              strawman for.

Writes study/RESULTS.md: a catch-rate table plus the raw pass/fail deltas.

Cost warning: this makes ~20 full evalgate runs (10 injections x 2 modes)
plus 2 baseline runs against the live Gemini API. See NOTES.md for the actual
per-run call-volume estimate before running this for real — on the Gemini
free tier this is expensive enough to need explicit planning, not a quick
`python study/run_study.py`.

Checkpointing: the baseline (run once) and each completed injection's row are
persisted to `--checkpoint` (default `study/study_checkpoint.json`) as soon as
they finish. Re-running with the same `--cases`/`--checkpoint` skips whatever
is already recorded and resumes from the next unfinished injection — this is
what makes a multi-day, quota-limited run possible instead of restarting from
scratch every time the free tier's daily cap is hit.

Quota safety: the runner (see evalgate.runner) never crashes a trial on a
rate-limit error or a transient server outage (e.g. Gemini 503s) — it
retries once, then records the trial as failed and moves on. Left
unchecked, a quota-exhausted or outage-hit stretch would silently produce
an all-trials-failed report that looks like legitimate (if boring) data
rather than a broken run. `_check_for_quota_failures` scans every trial in
a report for one of these transient failure reasons and raises
`QuotaExhausted` the moment one is found, so a tainted in-progress round is
never checkpointed or written to RESULTS.md — only fully clean rounds count
as done.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from examples.sec_agent.agent import build_agent
from examples.sec_agent.tools.edgar import EdgarClient, configure_client
from google.adk.models.lite_llm import LiteLlm

from evalgate.adapters.adk import ADKAdapter
from evalgate.compare import compare_runs, has_regressions
from evalgate.config import EvalGateConfig, load_config
from evalgate.loader import load_cases
from evalgate.runner import run_cases
from evalgate.schema import EvalCase, RunReport
from evalgate.stats import wilson_interval
from study.injections import INJECTIONS

_DETERMINISTIC_TYPES = {"contains", "regex", "numeric"}


class QuotaExhausted(RuntimeError):
    """Raised when a run's results are tainted by a rate-limit or transient
    server-outage failure — never checkpoint or report on data that
    includes one of these."""


def _check_for_quota_failures(report: RunReport, label: str) -> None:
    bad = [
        (cr.case_id, t.trial_idx)
        for cr in report.cases
        for t in cr.trials
        if t.failure_reason
        and any(
            marker in t.failure_reason
            for marker in (
                "RESOURCE_EXHAUSTED",
                "429",
                "RateLimitError",
                "rate_limit",
                "503",
                "UNAVAILABLE",
                "ServerError",
            )
        )
    ]
    if bad:
        raise QuotaExhausted(
            f"{label}: {len(bad)} trial(s) failed on a rate limit or "
            f"transient server outage (e.g. case {bad[0][0]!r} trial "
            f"{bad[0][1]}) — aborting before checkpointing tainted results."
        )


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"baseline": None, "injections": {}}


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def naive_cases(cases: list[EvalCase]) -> list[EvalCase]:
    """1 trial, deterministic scorers only — the eval suite a team without
    EvalGate might plausibly have shipped on their own.

    Raises ValueError if a case has no deterministic scorer at all — naive
    mode cannot express such a case, and silently giving it zero scorers
    would make score_trial() vacuously pass it (an empty `all()` is True).
    """
    result = []
    for case in cases:
        entries = [e for e in case.scoring if e.type in _DETERMINISTIC_TYPES]
        if not entries:
            raise ValueError(
                f"case {case.id!r} has no contains/regex/numeric scorer — "
                "naive mode cannot express it"
            )
        result.append(case.model_copy(update={"scoring": entries, "trials": 1}))
    return result


def case_result_rows(report: RunReport) -> list[dict]:
    """Shape a RunReport's cases into the dict rows compare_runs() expects —
    the same case_id/pass_rate/ci_low/ci_high keys store.py writes to
    case_results, computed without needing to touch SQLite."""
    rows = []
    for cr in report.cases:
        n = len(cr.trials)
        ci_low, ci_high = wilson_interval(cr.passes, n)
        rows.append(
            {
                "case_id": cr.case_id,
                "pass_rate": cr.pass_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return rows


def naive_regressions(before: RunReport, after: RunReport) -> list[str]:
    """Cases that passed (n=1) before and fail (n=1) after. No margin, no CI —
    that's the point of the naive strawman."""
    before_pass = {cr.case_id: cr.passes > 0 for cr in before.cases}
    after_pass = {cr.case_id: cr.passes > 0 for cr in after.cases}
    return [
        cid
        for cid, was_passing in before_pass.items()
        if was_passing and not after_pass.get(cid, True)
    ]


async def _build_adapter(
    build_kwargs: dict, cfg: EvalGateConfig, agent_model: str | None
) -> ADKAdapter:
    # Hardcoded replay, regardless of evalgate.toml — a study this expensive
    # must never accidentally hit the live network on top of its LLM cost.
    client = EdgarClient(
        mode="replay",
        fixtures_dir=cfg.edgar.fixtures_dir,
        user_agent=cfg.edgar.user_agent,
    )
    configure_client(client)
    # build_kwargs (e.g. the temperature_1_0 injection) takes precedence over
    # the config default — a dict-literal default + **build_kwargs would
    # collide with a duplicate "temperature" keyword instead of overriding it.
    kwargs = {"temperature": cfg.temperature, **build_kwargs}
    # No injection touches `model`, so it's safe to set last — this is what
    # routes the agent through Groq (via LiteLlm) instead of Gemini for the
    # whole study when --agent-model is given. LlmAgent.model accepts either
    # a bare string or a BaseLlm instance natively (see build_agent()'s
    # docstring), so wrapping only happens here, once.
    if agent_model is not None:
        kwargs["model"] = LiteLlm(model=agent_model)
    agent = build_agent(**kwargs)
    return ADKAdapter(agent)


async def _run_mode(
    cases: list[EvalCase], build_kwargs: dict, cfg: EvalGateConfig, agent_model: str | None
) -> RunReport:
    adapter = await _build_adapter(build_kwargs, cfg, agent_model)
    return await run_cases(cases, adapter, cfg)


async def run_study(
    cases_dir: str,
    output: str,
    checkpoint_path: Path,
    agent_model: str | None = None,
    max_concurrent_trials: int | None = None,
) -> None:
    cfg = load_config()
    cfg.validate()
    if max_concurrent_trials is not None:
        # Override only for this study run — evalgate.toml's default of 1 is
        # tuned for the Gemini free tier's 5 RPM cap (see the comment on that
        # line) and stays untouched for every other caller (`evalgate run`,
        # the pytest plugin). A paid-tier study run can afford real
        # concurrency instead of serializing every trial.
        cfg.max_concurrent_trials = max_concurrent_trials

    full_cases = load_cases(Path(cases_dir))
    naive_case_list = naive_cases(full_cases)

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("cases_dir") not in (None, str(cases_dir)):
        raise ValueError(
            f"checkpoint {checkpoint_path} was recorded for --cases "
            f"{checkpoint['cases_dir']!r}, not {cases_dir!r} — use a fresh "
            "--checkpoint path or the original --cases value."
        )
    checkpoint["cases_dir"] = str(cases_dir)
    # The baseline and every injection's "after" run must share one model —
    # mixing them would measure provider differences, not injection catches.
    # A resumed run on a different --agent-model would silently invalidate
    # every row already checkpointed, so it's a hard error, not a warning.
    if checkpoint.get("agent_model") not in (None, agent_model):
        raise ValueError(
            f"checkpoint {checkpoint_path} was recorded with --agent-model "
            f"{checkpoint['agent_model']!r}, not {agent_model!r} — the "
            "baseline and every injection must run on the same model. Use a "
            "fresh --checkpoint path to switch models."
        )
    checkpoint["agent_model"] = agent_model

    if checkpoint["baseline"] is not None:
        print("Resuming: baseline already checkpointed, skipping re-run.")
        full_before_rows = checkpoint["baseline"]["full_rows"]
        naive_before_pass = checkpoint["baseline"]["naive_pass"]
    else:
        print("Running full-mode baseline (before any injection)...")
        full_before = await _run_mode(full_cases, {}, cfg, agent_model)
        _check_for_quota_failures(full_before, "full-mode baseline")
        print("Running naive-mode baseline (before any injection)...")
        naive_before = await _run_mode(naive_case_list, {}, cfg, agent_model)
        _check_for_quota_failures(naive_before, "naive-mode baseline")

        full_before_rows = case_result_rows(full_before)
        naive_before_pass = {cr.case_id: cr.passes > 0 for cr in naive_before.cases}
        checkpoint["baseline"] = {"full_rows": full_before_rows, "naive_pass": naive_before_pass}
        save_checkpoint(checkpoint_path, checkpoint)

    done_numbers = {int(k) for k in checkpoint["injections"]}
    remaining = [inj for inj in INJECTIONS if inj.number not in done_numbers]
    if not remaining:
        print("All injections already checkpointed.")
    for injection in remaining:
        print(f"\n=== Injection #{injection.number}: {injection.id} ===")
        with injection.apply():
            print("  full mode...")
            full_after = await _run_mode(full_cases, injection.build_kwargs, cfg, agent_model)
            _check_for_quota_failures(full_after, f"injection #{injection.number} full mode")
            print("  naive mode...")
            naive_after = await _run_mode(
                naive_case_list, injection.build_kwargs, cfg, agent_model
            )
            _check_for_quota_failures(naive_after, f"injection #{injection.number} naive mode")

        full_comparisons = compare_runs(
            full_before_rows, case_result_rows(full_after), cfg.regression_margin
        )
        full_caught = has_regressions(full_comparisons)
        after_pass = {cr.case_id: cr.passes > 0 for cr in naive_after.cases}
        naive_caught = bool(
            [
                cid
                for cid, was_passing in naive_before_pass.items()
                if was_passing and not after_pass.get(cid, True)
            ]
        )

        row = {
            "number": injection.number,
            "id": injection.id,
            "description": injection.description,
            "full_caught": full_caught,
            "naive_caught": naive_caught,
        }
        print(f"  full mode caught: {full_caught}   naive mode caught: {naive_caught}")

        checkpoint["injections"][str(injection.number)] = row
        save_checkpoint(checkpoint_path, checkpoint)

    rows = [checkpoint["injections"][str(inj.number)] for inj in INJECTIONS]
    if len(rows) < len(INJECTIONS):
        print(
            f"\n{len(rows)}/{len(INJECTIONS)} injections checkpointed so far — "
            f"rerun the same command tomorrow (or once quota resets) to continue."
        )
        return

    write_results(rows, output)


def write_results(rows: list[dict], output: str) -> None:
    full_hits = sum(r["full_caught"] for r in rows)
    naive_hits = sum(r["naive_caught"] for r in rows)
    lines = [
        "# Regression-Injection Study Results",
        "",
        f"EvalGate (full mode) caught **{full_hits}/{len(rows)}**. "
        f"Naive mode (1 trial, exact-match-only) caught **{naive_hits}/{len(rows)}**.",
        "",
        "| # | Injection | EvalGate (full) | Naive (1 trial) |",
        "|---|-----------|:---:|:---:|",
    ]
    for r in rows:
        full_mark = "caught" if r["full_caught"] else "missed"
        naive_mark = "caught" if r["naive_caught"] else "missed"
        lines.append(f"| {r['number']} | {r['description']} | {full_mark} | {naive_mark} |")
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="examples/sec_agent/cases")
    parser.add_argument("--output", default="study/RESULTS.md")
    parser.add_argument("--checkpoint", default="study/study_checkpoint.json")
    parser.add_argument(
        "--agent-model",
        default=None,
        help=(
            "litellm model string routing the agent through a non-Gemini "
            "provider via LiteLlm, e.g. 'groq/llama-3.1-8b-instant'. "
            "Default (unset) keeps the agent on its normal Gemini model. "
            "The judge always stays on evalgate.toml's judge_model — this "
            "flag only affects the agent under test."
        ),
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=None,
        help="Override evalgate.toml's max_concurrent_trials for this run only.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            run_study(
                args.cases,
                args.output,
                Path(args.checkpoint),
                args.agent_model,
                args.max_concurrent_trials,
            )
        )
    except QuotaExhausted as exc:
        print(f"\nStopped — quota exhausted: {exc}")
        print(f"Already-completed rounds are safely checkpointed in {args.checkpoint}.")
        print("Rerun the same command once quota resets to resume.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
