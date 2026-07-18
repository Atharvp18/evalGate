"""Unit tests for store.py — schema creation, save/load round-trip."""

from __future__ import annotations

import json

from evalgate.config import EvalGateConfig
from evalgate.schema import CaseResult, EvalCase, RunReport, TrialResult
from evalgate.store import connect, latest_run_id, load_run, save_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case(case_id: str = "case_a", pass_rate_min: float = 0.75) -> EvalCase:
    return EvalCase(
        id=case_id,
        description="test",
        input="q",
        expected={"contains": ["x"]},
        scoring=[{"type": "contains"}],
        thresholds={"pass_rate_min": pass_rate_min},
    )


def _report() -> RunReport:
    trials = [
        TrialResult(
            trial_idx=0,
            passed=True,
            final_text="answer with x",
            scores=[{"scorer_type": "contains", "passed": True, "detail": "ok", "extra": {}}],
            tool_calls=[{"name": "lookup_cik", "args": {"ticker": "NVDA"}}],
            latency_ms=123.0,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        ),
        TrialResult(
            trial_idx=1,
            passed=False,
            failure_reason="contains: missing: ['x']",
            final_text="wrong answer",
            scores=[{"scorer_type": "contains", "passed": False, "detail": "missing", "extra": {}}],
            judge_prompt="rubric prompt",
            judge_response='{"pass": false}',
        ),
    ]
    return RunReport(cases=[CaseResult(case_id="case_a", trials=trials)])


# ---------------------------------------------------------------------------
# Tests (in-memory SQLite — no files, no network)
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip():
    conn = connect(":memory:")
    run_id = save_run(conn, _report(), [_case()], EvalGateConfig())

    data = load_run(conn, run_id)
    assert data["run"]["agent_name"] == "sec_agent"
    assert json.loads(data["run"]["config_json"])["trials"] == 6

    (cr,) = data["cases"]
    assert cr["case_id"] == "case_a"
    assert cr["trials"] == 2
    assert cr["passes"] == 1
    assert cr["pass_rate"] == 0.5
    assert 0.0 <= cr["ci_low"] < 0.5 < cr["ci_high"] <= 1.0
    assert cr["passed_threshold"] == 0  # 0.5 < pass_rate_min 0.75

    t0, t1 = cr["trial_rows"]
    assert t0["passed"] == 1
    assert json.loads(t0["tool_calls_json"])[0]["name"] == "lookup_cik"
    assert t1["passed"] == 0
    assert t1["failure_reason"] == "contains: missing: ['x']"
    assert t1["judge_prompt"] == "rubric prompt"


def test_threshold_pass_when_rate_meets_min():
    conn = connect(":memory:")
    run_id = save_run(conn, _report(), [_case(pass_rate_min=0.5)], EvalGateConfig())
    (cr,) = load_run(conn, run_id)["cases"]
    assert cr["passed_threshold"] == 1


def test_latest_run_id_increments():
    conn = connect(":memory:")
    assert latest_run_id(conn) is None
    first = save_run(conn, _report(), [_case()], EvalGateConfig())
    second = save_run(conn, _report(), [_case()], EvalGateConfig())
    assert second > first
    assert latest_run_id(conn) == second


def test_load_missing_run_raises():
    conn = connect(":memory:")
    try:
        load_run(conn, 999)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "999" in str(e)


def test_flaky_flag_persisted():
    # 1/2 passes at n=2 → very wide CI → flaky with default width 0.5.
    conn = connect(":memory:")
    run_id = save_run(conn, _report(), [_case()], EvalGateConfig())
    (cr,) = load_run(conn, run_id)["cases"]
    assert cr["flaky"] == 1
