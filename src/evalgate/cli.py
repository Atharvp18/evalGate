"""EvalGate CLI — entry point for all user-facing commands."""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from pathlib import Path

import typer

app = typer.Typer(
    name="evalgate",
    help="Evaluation and regression-testing framework for LLM agent systems.",
    no_args_is_help=True,
)

baseline_app = typer.Typer(help="Manage baselines.")
app.add_typer(baseline_app, name="baseline")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=level)


def _build_sec_agent_adapter(config: object) -> object:
    """Construct the sec_agent and wrap it in an ADKAdapter.

    The EDGAR client is configured in replay mode for eval runs so no
    network calls are made. Live mode is reserved for the interactive REPL
    and the nightly smoke test.
    """
    import sys

    # examples.sec_agent is not an installed package — it lives at the repo root.
    # Add the repo root to sys.path so it's importable when evalgate runs as an
    # installed entry point (which does not add cwd to sys.path automatically).
    # cli.py is at src/evalgate/cli.py, so three .parent calls reach the repo root.
    _repo_root = Path(__file__).resolve().parent.parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from dotenv import load_dotenv
    from examples.sec_agent.agent import build_agent
    from examples.sec_agent.tools.edgar import EdgarClient, configure_client

    from evalgate.adapters.adk import ADKAdapter

    load_dotenv(override=False)

    edgar_cfg = config.edgar  # type: ignore[attr-defined]
    client = EdgarClient(
        mode=edgar_cfg.mode,
        fixtures_dir=edgar_cfg.fixtures_dir,
        user_agent=edgar_cfg.user_agent,
        requests_per_second=edgar_cfg.requests_per_second,
    )
    configure_client(client)
    agent = build_agent()
    return ADKAdapter(agent)


def _print_run_report(report: object, verbose: bool = False) -> None:  # type: ignore[type-arg]
    """Print a human-readable summary table for a RunReport."""
    from evalgate.schema import RunReport

    assert isinstance(report, RunReport)

    header = f"{'CASE':<35} {'TRIALS':>6} {'PASS':>5} {'RATE':>6} {'P50 ms':>8}  SAMPLE OUTPUT"
    typer.echo("\n" + "=" * len(header))
    typer.echo(header)
    typer.echo("=" * len(header))

    for cr in report.cases:
        n = len(cr.trials)
        p50 = cr.p50_latency_ms
        sample = ""
        # Pick the first successful trial's output as a sample.
        for t in cr.trials:
            if t.passed and t.final_text:
                sample = t.final_text[:70].replace("\n", " ")
                break
        typer.echo(
            f"{cr.case_id:<35} {n:>6} {cr.passes:>5} {cr.pass_rate:>5.0%} {p50:>8.0f}  {sample}"
        )

    typer.echo("=" * len(header))

    # Failure details: one line per failed trial so the user sees which scorer
    # failed and why without needing the DB (full detail lands there in Phase 5).
    for cr in report.cases:
        for t in cr.trials:
            if not t.passed and t.failure_reason:
                reason = t.failure_reason.replace("\n", " ")
                if not verbose:
                    reason = reason[:160]
                typer.echo(f"  FAIL {cr.case_id} trial {t.trial_idx}: {reason}")

    # Cost / token summary.
    in_tok = report.total_input_tokens
    out_tok = report.total_output_tokens
    cost = report.total_cost_usd
    all_lats = [t.latency_ms for cr in report.cases for t in cr.trials if t.latency_ms > 0]
    if len(all_lats) >= 2:
        p95 = statistics.quantiles(all_lats, n=20)[18]
    else:
        p95 = all_lats[0] if all_lats else 0
    typer.echo(
        f"\nTokens: {in_tok:,} in / {out_tok:,} out  |  "
        f"Est. cost: ${cost:.4f}  |  "
        f"p95 latency: {p95:.0f} ms"
    )


@app.command()
def run(
    cases: str = typer.Option(..., "--cases", help="Directory containing eval case YAML files."),
    trials: int = typer.Option(None, "--trials", help="Override trial count for all cases."),
    db: str = typer.Option(None, "--db", help="SQLite DB path (overrides evalgate.toml)."),
    agent: str = typer.Option("sec_agent", "--agent", help="Agent to evaluate."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run eval cases against the agent and store results."""
    _setup_logging(verbose)

    from evalgate.config import load_config
    from evalgate.loader import load_cases
    from evalgate.runner import run_cases
    from evalgate.store import connect, save_run

    cfg = load_config()
    try:
        cfg.validate()
    except ValueError as e:
        typer.echo(f"Config error: {e}", err=True)
        raise typer.Exit(2) from None  # noqa: B904

    if db:
        cfg.db_path = db

    typer.echo(f"Loading cases from: {cases}")
    try:
        eval_cases = load_cases(Path(cases))
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(2) from None  # noqa: B904
    typer.echo(f"Found {len(eval_cases)} case(s): {', '.join(c.id for c in eval_cases)}")

    if agent != "sec_agent":
        typer.echo(f"Unknown agent {agent!r}. Only 'sec_agent' is supported.", err=True)
        raise typer.Exit(2)

    typer.echo("Building agent adapter (replay mode)...")
    try:
        adapter = _build_sec_agent_adapter(cfg)
    except Exception as e:
        typer.echo(f"Failed to build agent: {e}", err=True)
        raise typer.Exit(2) from None  # noqa: B904

    typer.echo(f"Running {len(eval_cases)} case(s){f' × {trials} trials' if trials else ''}...\n")
    try:
        report = asyncio.run(run_cases(eval_cases, adapter, cfg, trials_override=trials))
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")
        raise typer.Exit(1) from None  # noqa: B904

    _print_run_report(report, verbose=verbose)

    conn = connect(cfg.db_path)
    new_run_id = save_run(conn, report, eval_cases, cfg)
    conn.close()
    typer.echo(f"\nSaved as run {new_run_id}. View with: evalgate report --run-id {new_run_id}")


@app.command()
def report(
    run_id: int = typer.Option(None, "--run-id", help="Specific run ID to report."),
    latest: bool = typer.Option(False, "--latest", help="Report on the latest run."),
) -> None:
    """Print a report for a stored run."""
    from evalgate.config import load_config
    from evalgate.store import connect, latest_run_id, load_run

    cfg = load_config()
    conn = connect(cfg.db_path)

    if run_id is None:
        run_id = latest_run_id(conn)
        if run_id is None:
            typer.echo("No runs stored yet. Run `evalgate run` first.", err=True)
            raise typer.Exit(2)

    try:
        data = load_run(conn, run_id)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(2) from None  # noqa: B904
    finally:
        conn.close()

    run = data["run"]
    typer.echo(
        f"Run {run['id']}  |  {run['started_at']}  |  agent: {run['agent_name']}"
        f"  |  git: {(run['git_sha'] or '?')[:8]} ({run['git_branch'] or '?'})"
    )

    header = (
        f"{'CASE':<35} {'TRIALS':>6} {'PASS':>5} {'RATE':>6} "
        f"{'CI 95%':>14} {'FLAKY':>6} {'THRESH':>7}"
    )
    typer.echo("=" * len(header))
    typer.echo(header)
    typer.echo("=" * len(header))
    for cr in data["cases"]:
        ci = f"[{cr['ci_low']:.2f}, {cr['ci_high']:.2f}]"
        typer.echo(
            f"{cr['case_id']:<35} {cr['trials']:>6} {cr['passes']:>5} "
            f"{cr['pass_rate']:>5.0%} {ci:>14} "
            f"{'yes' if cr['flaky'] else '':>6} "
            f"{'ok' if cr['passed_threshold'] else 'FAIL':>7}"
        )
    typer.echo("=" * len(header))

    # Per-trial scorer details for failed trials.
    for cr in data["cases"]:
        for t in cr["trial_rows"]:
            if t["passed"]:
                continue
            typer.echo(f"\n{cr['case_id']} trial {t['trial_idx']} FAILED:")
            for s in json.loads(t["scores_json"]):
                mark = "pass" if s["passed"] else "FAIL"
                typer.echo(f"  [{mark}] {s['scorer_type']}: {s['detail'][:200]}")
            if not json.loads(t["scores_json"]) and t["failure_reason"]:
                typer.echo(f"  {t['failure_reason'][:300]}")

    total_cost = sum(t["cost_usd"] or 0 for cr in data["cases"] for t in cr["trial_rows"])
    in_tok = sum(t["input_tokens"] or 0 for cr in data["cases"] for t in cr["trial_rows"])
    out_tok = sum(t["output_tokens"] or 0 for cr in data["cases"] for t in cr["trial_rows"])
    lats = sorted(
        t["latency_ms"] for cr in data["cases"] for t in cr["trial_rows"] if t["latency_ms"]
    )
    p50 = statistics.median(lats) if lats else 0
    p95 = statistics.quantiles(lats, n=20)[18] if len(lats) >= 2 else (lats[0] if lats else 0)
    typer.echo(
        f"\nTokens: {in_tok:,} in / {out_tok:,} out  |  Est. cost: ${total_cost:.4f}"
        f"  |  latency p50: {p50:.0f} ms, p95: {p95:.0f} ms"
    )


@baseline_app.command("save")
def baseline_save(
    name: str = typer.Option("main", "--name", help="Baseline name."),
    run_id: int = typer.Option(None, "--run-id", help="Run ID to save (default: latest)."),
) -> None:
    """Save a run as a named baseline."""
    typer.echo("evalgate baseline save — not implemented yet (Phase 6)")


@app.command()
def compare(
    baseline: str = typer.Option(..., "--baseline", help="Baseline name to compare against."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Compare the latest run against a baseline and exit 1 if regressions found."""
    typer.echo("evalgate compare — not implemented yet (Phase 6)")


@app.command()
def calibrate(
    labels: str = typer.Option(..., "--labels", help="Path to human_labels.csv."),
    judge_model: str = typer.Option(None, "--judge-model", help="Override judge model string."),
    export: bool = typer.Option(False, "--export", help="Export unlabelled CSV for human review."),
) -> None:
    """Compute judge-vs-human agreement (Cohen's kappa) from labelled trials."""
    typer.echo("evalgate calibrate — not implemented yet (Phase 7)")


@app.command(name="mine-trace")
def mine_trace(
    run_id: int = typer.Option(..., "--run-id", help="Run ID containing the trial."),
    trial_id: int = typer.Option(..., "--trial-id", help="Trial ID to mine."),
    output: str = typer.Option(..., "-o", help="Output YAML path for the new case."),
) -> None:
    """Generate a draft eval case YAML from a stored failed trial."""
    typer.echo("evalgate mine-trace — not implemented yet (Phase 8)")
