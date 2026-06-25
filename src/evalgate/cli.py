"""EvalGate CLI — entry point for all user-facing commands."""

import typer

app = typer.Typer(
    name="evalgate",
    help="Evaluation and regression-testing framework for LLM agent systems.",
    no_args_is_help=True,
)

baseline_app = typer.Typer(help="Manage baselines.")
app.add_typer(baseline_app, name="baseline")


@app.command()
def run(
    cases: str = typer.Option(..., help="Directory containing eval case YAML files."),
    trials: int = typer.Option(None, help="Override default trial count."),
    db: str = typer.Option(None, help="SQLite DB path (overrides evalgate.toml)."),
    agent: str = typer.Option("sec_agent", help="Agent to evaluate."),
) -> None:
    """Run eval cases against the agent and store results."""
    typer.echo("evalgate run — not implemented yet (Phase 3)")


@app.command()
def report(
    run_id: int = typer.Option(None, help="Specific run ID to report."),
    latest: bool = typer.Option(False, "--latest", help="Report on the latest run."),
) -> None:
    """Print a report for a stored run."""
    typer.echo("evalgate report — not implemented yet (Phase 5)")


@baseline_app.command("save")
def baseline_save(
    name: str = typer.Option("main", help="Baseline name."),
    run_id: int = typer.Option(None, help="Run ID to save as baseline (defaults to latest)."),
) -> None:
    """Save a run as a named baseline."""
    typer.echo("evalgate baseline save — not implemented yet (Phase 6)")


@app.command()
def compare(
    baseline: str = typer.Option(..., help="Baseline name to compare against."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Compare the latest run against a baseline and exit 1 if regressions found."""
    typer.echo("evalgate compare — not implemented yet (Phase 6)")


@app.command()
def calibrate(
    labels: str = typer.Option(..., help="Path to human_labels.csv."),
    judge_model: str = typer.Option(None, help="Override judge model string."),
    export: bool = typer.Option(False, "--export", help="Export unlabelled CSV for human review."),
) -> None:
    """Compute judge-vs-human agreement (Cohen's kappa) from labelled trials."""
    typer.echo("evalgate calibrate — not implemented yet (Phase 7)")


@app.command(name="mine-trace")
def mine_trace(
    run_id: int = typer.Option(..., help="Run ID containing the trial."),
    trial_id: int = typer.Option(..., help="Trial ID to mine."),
    output: str = typer.Option(..., "-o", help="Output YAML path for the new case."),
) -> None:
    """Generate a draft eval case YAML from a stored failed trial."""
    typer.echo("evalgate mine-trace — not implemented yet (Phase 8)")
