"""EvalGate CLI — entry point for all user-facing commands."""

from __future__ import annotations

import asyncio
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


@app.command()
def report(
    run_id: int = typer.Option(None, "--run-id", help="Specific run ID to report."),
    latest: bool = typer.Option(False, "--latest", help="Report on the latest run."),
) -> None:
    """Print a report for a stored run."""
    typer.echo("evalgate report — not implemented yet (Phase 5)")


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
