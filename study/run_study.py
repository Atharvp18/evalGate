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
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from examples.sec_agent.agent import build_agent
from examples.sec_agent.tools.edgar import EdgarClient, configure_client

from evalgate.adapters.adk import ADKAdapter
from evalgate.compare import compare_runs, has_regressions
from evalgate.config import EvalGateConfig, load_config
from evalgate.loader import load_cases
from evalgate.runner import run_cases
from evalgate.schema import EvalCase, RunReport
from evalgate.stats import wilson_interval
from study.injections import INJECTIONS

_DETERMINISTIC_TYPES = {"contains", "regex", "numeric"}


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


async def _build_adapter(build_kwargs: dict, cfg: EvalGateConfig) -> ADKAdapter:
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
    agent = build_agent(**kwargs)
    return ADKAdapter(agent)


async def _run_mode(cases: list[EvalCase], build_kwargs: dict, cfg: EvalGateConfig) -> RunReport:
    adapter = await _build_adapter(build_kwargs, cfg)
    return await run_cases(cases, adapter, cfg)


async def run_study(cases_dir: str, output: str) -> None:
    cfg = load_config()
    cfg.validate()

    full_cases = load_cases(Path(cases_dir))
    naive_case_list = naive_cases(full_cases)

    print("Running full-mode baseline (before any injection)...")
    full_before = await _run_mode(full_cases, {}, cfg)
    print("Running naive-mode baseline (before any injection)...")
    naive_before = await _run_mode(naive_case_list, {}, cfg)

    rows = []
    for injection in INJECTIONS:
        print(f"\n=== Injection #{injection.number}: {injection.id} ===")
        with injection.apply():
            print("  full mode...")
            full_after = await _run_mode(full_cases, injection.build_kwargs, cfg)
            print("  naive mode...")
            naive_after = await _run_mode(naive_case_list, injection.build_kwargs, cfg)

        full_comparisons = compare_runs(
            case_result_rows(full_before), case_result_rows(full_after), cfg.regression_margin
        )
        full_caught = has_regressions(full_comparisons)
        naive_caught = bool(naive_regressions(naive_before, naive_after))

        rows.append(
            {
                "number": injection.number,
                "id": injection.id,
                "description": injection.description,
                "full_caught": full_caught,
                "naive_caught": naive_caught,
            }
        )
        print(f"  full mode caught: {full_caught}   naive mode caught: {naive_caught}")

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
    args = parser.parse_args()
    asyncio.run(run_study(args.cases, args.output))


if __name__ == "__main__":
    main()
