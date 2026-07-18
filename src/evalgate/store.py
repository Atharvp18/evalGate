"""SQLite persistence — hand-written DDL, writers, and readers for runs, trials, and baselines."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evalgate.config import EvalGateConfig
from evalgate.schema import EvalCase, RunReport
from evalgate.stats import is_flaky, wilson_interval

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  git_sha TEXT,
  git_branch TEXT,
  config_json TEXT NOT NULL,
  agent_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_results (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  case_id TEXT NOT NULL,
  trials INTEGER NOT NULL,
  passes INTEGER NOT NULL,
  pass_rate REAL NOT NULL,
  ci_low REAL NOT NULL,
  ci_high REAL NOT NULL,
  flaky INTEGER NOT NULL,
  passed_threshold INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
  id INTEGER PRIMARY KEY,
  case_result_id INTEGER NOT NULL REFERENCES case_results(id),
  trial_idx INTEGER NOT NULL,
  passed INTEGER NOT NULL,
  failure_reason TEXT,
  final_text TEXT,
  scores_json TEXT NOT NULL,
  tool_calls_json TEXT NOT NULL,
  raw_events_json TEXT,
  latency_ms REAL,
  input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
  judge_prompt TEXT, judge_response TEXT
);
CREATE TABLE IF NOT EXISTS baselines (
  name TEXT PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  saved_at TEXT NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the evalgate DB and ensure the schema exists."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    return conn


def _git_info() -> tuple[str | None, str | None]:
    """Best-effort (sha, branch); None outside a git repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return (sha or None, branch or None)
    except OSError:
        return (None, None)


def save_run(
    conn: sqlite3.Connection,
    report: RunReport,
    cases: list[EvalCase],
    config: EvalGateConfig,
) -> int:
    """Persist a full run (run → case_results → trials) in one transaction.

    Returns the new run id. `cases` supplies each case's pass_rate_min so the
    passed_threshold verdict can be stored alongside the stats.
    """
    thresholds = {c.id: c.thresholds.pass_rate_min for c in cases}
    sha, branch = _git_info()

    with conn:  # one transaction for the whole run
        cur = conn.execute(
            "INSERT INTO runs (started_at, git_sha, git_branch, config_json, agent_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                sha,
                branch,
                json.dumps(config, default=lambda o: o.__dict__),
                report.agent_name,
            ),
        )
        run_id = cur.lastrowid
        assert run_id is not None

        for cr in report.cases:
            n = len(cr.trials)
            ci_low, ci_high = wilson_interval(cr.passes, n)
            cur = conn.execute(
                "INSERT INTO case_results "
                "(run_id, case_id, trials, passes, pass_rate, ci_low, ci_high, flaky, "
                "passed_threshold) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    cr.case_id,
                    n,
                    cr.passes,
                    cr.pass_rate,
                    ci_low,
                    ci_high,
                    int(is_flaky(cr.passes, n, config.flaky_ci_width)),
                    int(cr.pass_rate >= thresholds.get(cr.case_id, 0.75)),
                ),
            )
            case_result_id = cur.lastrowid

            conn.executemany(
                "INSERT INTO trials "
                "(case_result_id, trial_idx, passed, failure_reason, final_text, scores_json, "
                "tool_calls_json, raw_events_json, latency_ms, input_tokens, output_tokens, "
                "cost_usd, judge_prompt, judge_response) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        case_result_id,
                        t.trial_idx,
                        int(t.passed),
                        t.failure_reason,
                        t.final_text,
                        json.dumps(t.scores),
                        json.dumps(t.tool_calls),
                        json.dumps(t.raw_events),
                        t.latency_ms,
                        t.input_tokens,
                        t.output_tokens,
                        t.cost_usd,
                        t.judge_prompt,
                        t.judge_response,
                    )
                    for t in cr.trials
                ],
            )

    return run_id


def latest_run_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT MAX(id) AS id FROM runs").fetchone()
    return row["id"]


def load_run(conn: sqlite3.Connection, run_id: int) -> dict:
    """Load one run with its case_results and trials as plain dicts.

    Returns {"run": {...}, "cases": [{..., "trial_rows": [{...}]}]}. The key
    is "trial_rows" (not "trials") because the case_results row already has a
    "trials" column holding the count. Raises ValueError if the run id does
    not exist.
    """
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise ValueError(f"run {run_id} not found in the database")

    cases = []
    for cr in conn.execute(
        "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id", (run_id,)
    ):
        trials = [
            dict(t)
            for t in conn.execute(
                "SELECT * FROM trials WHERE case_result_id = ? ORDER BY trial_idx", (cr["id"],)
            )
        ]
        cases.append({**dict(cr), "trial_rows": trials})

    return {"run": dict(run), "cases": cases}
