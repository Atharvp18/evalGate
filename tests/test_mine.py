"""Unit tests for mine.py — trace mining from a stored trial into a draft case YAML.

No LLM calls: uses the same in-memory SQLite + prebuilt TrialResult fixtures
convention as test_store.py, plus a temp cases dir for input-text recovery.
"""

from __future__ import annotations

import yaml

from evalgate.config import EvalGateConfig
from evalgate.mine import TrialNotFoundError, mine_trace
from evalgate.schema import CaseResult, EvalCase, RunReport, TrialResult
from evalgate.store import connect, save_run

CASE_YAML = """\
id: nvda_latest_revenue
description: Agent retrieves Nvidia's most recent quarterly revenue
input: "What was Nvidia's revenue in its most recent quarter?"
scoring:
  - type: contains
expected:
  contains: ["Nvidia"]
"""


def _case() -> EvalCase:
    return EvalCase(
        id="nvda_latest_revenue",
        description="test",
        input="What was Nvidia's revenue in its most recent quarter?",
        expected={"contains": ["Nvidia"]},
        scoring=[{"type": "contains"}],
    )


def _report() -> RunReport:
    trials = [
        TrialResult(
            trial_idx=0,
            passed=False,
            failure_reason="contains: missing: ['Nvidia']",
            final_text="Some other company made money.",
            scores=[{"scorer_type": "contains", "passed": False, "detail": "x", "extra": {}}],
            tool_calls=[
                {"name": "transfer_to_agent", "args": {"agent_name": "retrieval_agent"}},
                {"name": "lookup_cik", "args": {"ticker": "NVDA"}},
                {"name": "get_company_facts", "args": {"cik": "0001045810"}},
            ],
        ),
    ]
    return RunReport(cases=[CaseResult(case_id="nvda_latest_revenue", trials=trials)])


def _seed(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "nvda_latest_revenue.yaml").write_text(CASE_YAML)

    conn = connect(":memory:")
    run_id = save_run(conn, _report(), [_case()], EvalGateConfig())
    return conn, run_id, cases_dir


def test_mine_trace_writes_yaml_with_recovered_input(tmp_path):
    conn, run_id, cases_dir = _seed(tmp_path)
    out = tmp_path / "mined.yaml"

    path = mine_trace(conn, run_id, trial_id=1, output_path=out, cases_dir=cases_dir)
    assert path == str(out)

    text = out.read_text()
    assert text.startswith("# mined from run")
    assert "original failure: contains: missing: ['Nvidia']" in text

    draft = yaml.safe_load(text)
    assert draft["id"] == f"mined_nvda_latest_revenue_{run_id}_1"
    assert draft["input"] == "What was Nvidia's revenue in its most recent quarter?"
    assert draft["pass_policy"] == "all"


def test_mine_trace_trajectory_excludes_transfer_calls(tmp_path):
    conn, run_id, cases_dir = _seed(tmp_path)
    out = tmp_path / "mined.yaml"
    mine_trace(conn, run_id, trial_id=1, output_path=out, cases_dir=cases_dir)

    draft = yaml.safe_load(out.read_text())
    tools = [step["tool"] for step in draft["expected"]["trajectory"]]
    assert tools == ["lookup_cik", "get_company_facts"]
    assert "transfer_to_agent" not in tools
    assert {"type": "trajectory"} in draft["scoring"]


def test_mine_trace_always_includes_todo_judge_rubric(tmp_path):
    conn, run_id, cases_dir = _seed(tmp_path)
    out = tmp_path / "mined.yaml"
    mine_trace(conn, run_id, trial_id=1, output_path=out, cases_dir=cases_dir)

    draft = yaml.safe_load(out.read_text())
    judge_entries = [s for s in draft["scoring"] if s["type"] == "judge"]
    assert len(judge_entries) == 1
    assert judge_entries[0]["rubric"].startswith("TODO")


def test_mine_trace_missing_trial_raises(tmp_path):
    conn, run_id, cases_dir = _seed(tmp_path)
    try:
        mine_trace(conn, run_id, trial_id=999, output_path=tmp_path / "x.yaml", cases_dir=cases_dir)
        raise AssertionError("expected TrialNotFoundError")
    except TrialNotFoundError as e:
        assert "999" in str(e)


def test_mine_trace_wrong_run_id_raises(tmp_path):
    conn, run_id, cases_dir = _seed(tmp_path)
    try:
        mine_trace(
            conn, run_id + 1, trial_id=1, output_path=tmp_path / "x.yaml", cases_dir=cases_dir
        )
        raise AssertionError("expected TrialNotFoundError")
    except TrialNotFoundError:
        pass


def test_mine_trace_creates_parent_dirs(tmp_path):
    conn, run_id, cases_dir = _seed(tmp_path)
    out = tmp_path / "nested" / "dir" / "mined.yaml"
    mine_trace(conn, run_id, trial_id=1, output_path=out, cases_dir=cases_dir)
    assert out.exists()
