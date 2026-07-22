"""Trace mining — converts a stored failed trial into a draft eval case YAML.

The point is the dev-loop story: a production failure gets turned into a
regression test with one command. The generated YAML is a starting point, not
a finished case — the trajectory reflects what the trial ACTUALLY did (not
what it should have done) and the rubric is a TODO placeholder, because both
require human judgment about what "correct" looks like.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Sub-agent hand-offs are ADK plumbing, not something a human would assert on
# in a trajectory scorer — the mined trajectory keeps only the real tool calls.
_NON_TOOL_CALLS = frozenset({"transfer_to_agent"})


class TrialNotFoundError(ValueError):
    """Raised when the (run_id, trial_id) pair does not exist in the DB."""


def _load_trial_row(conn: sqlite3.Connection, run_id: int, trial_id: int) -> dict:
    row = conn.execute(
        "SELECT t.*, cr.case_id FROM trials t "
        "JOIN case_results cr ON t.case_result_id = cr.id "
        "WHERE t.id = ? AND cr.run_id = ?",
        (trial_id, run_id),
    ).fetchone()
    if row is None:
        raise TrialNotFoundError(f"No trial {trial_id} found in run {run_id}")
    return dict(row)


def _lookup_case_input(cases_dir: str | Path, case_id: str) -> str:
    """Recover the original question from the case's own YAML file.

    Trials don't store the input query directly (only the agent's output), but
    every trial for a case was run against the same fixed `input` field in its
    YAML — so re-loading the case file is the exact original text, not a
    reconstruction.
    """
    from evalgate.loader import load_cases

    cases = load_cases(cases_dir)
    for c in cases:
        if c.id == case_id:
            return c.input
    raise ValueError(f"case {case_id!r} not found in {cases_dir} — cannot recover its input text")


def mine_trace(
    conn: sqlite3.Connection,
    run_id: int,
    trial_id: int,
    output_path: str | Path,
    cases_dir: str | Path = "examples/sec_agent/cases",
) -> str:
    """Generate a draft eval case YAML from a stored trial. Returns the written path."""
    row = _load_trial_row(conn, run_id, trial_id)
    case_id = row["case_id"]
    input_text = _lookup_case_input(cases_dir, case_id)

    tool_calls = json.loads(row["tool_calls_json"])
    trajectory = [{"tool": tc["name"]} for tc in tool_calls if tc["name"] not in _NON_TOOL_CALLS]

    scoring: list[dict] = []
    expected: dict = {}
    if trajectory:
        expected["trajectory"] = trajectory
        scoring.append({"type": "trajectory"})
    scoring.append(
        {
            "type": "judge",
            "rubric": "TODO: write a rubric describing what a correct answer looks like",
        }
    )

    draft = {
        "id": f"mined_{case_id}_{run_id}_{trial_id}",
        "description": f"TODO: describe this case — mined from a failing trial of {case_id!r}",
        "input": input_text,
        "trials": 4,
        "expected": expected,
        "scoring": scoring,
        "pass_policy": "all",
    }

    header = (
        f"# mined from run {run_id} trial {trial_id} on "
        f"{datetime.now(UTC).date().isoformat()} — review before committing\n"
        f"# original failure: {row['failure_reason'] or 'a scorer failed, see the run report'}\n"
    )
    body = yaml.safe_dump(draft, sort_keys=False, default_flow_style=False, allow_unicode=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body, encoding="utf-8")
    return str(out_path)
