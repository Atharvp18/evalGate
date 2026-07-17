"""Unit tests for the scorers — number normalizer, contains/regex/numeric,
trajectory subsequence matching, and judge JSON parsing (stubbed, no LLM calls)."""

from __future__ import annotations

import pytest

from evalgate.adapter import AgentRunResult, ToolCall
from evalgate.schema import EvalCase
from evalgate.scorers.deterministic import (
    extract_numbers,
    score_contains,
    score_numeric,
    score_regex,
)
from evalgate.scorers.judge import _parse_judge_json
from evalgate.scorers.trajectory import score_trajectory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case(**overrides) -> EvalCase:
    base = {
        "id": "test_case",
        "description": "test",
        "input": "What is X?",
        "scoring": [{"type": "judge", "rubric": "correct"}],
    }
    base.update(overrides)
    return EvalCase(**base)


def _result(text: str = "", tool_calls: list[ToolCall] | None = None) -> AgentRunResult:
    return AgentRunResult(final_text=text, tool_calls=tool_calls or [])


# ---------------------------------------------------------------------------
# Number normalizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$60.9 billion", 60_900_000_000),
        ("60,922 million", 60_922_000_000),
        ("60922000000", 60_922_000_000),
        ("revenue grew to 60.92B", 60_920_000_000),
        ("$81,615 million", 81_615_000_000),
        ("1.2 trillion in assets", 1_200_000_000_000),
        ("about 500 thousand units", 500_000),
        ("-3.5 billion loss", -3_500_000_000),
    ],
)
def test_extract_numbers_scales(text: str, expected: float):
    assert expected in extract_numbers(text)


def test_extract_numbers_skips_percentages():
    # 12% must not appear; 60.9 billion must.
    nums = extract_numbers("revenue grew 12% to $60.9 billion")
    assert 12 not in nums
    assert 60_900_000_000 in nums


def test_extract_numbers_plain_and_comma_mix():
    nums = extract_numbers("Q1 revenue was 81,615,000,000 dollars")
    assert 81_615_000_000 in nums


def test_extract_numbers_no_false_scale_from_words():
    # "b" in "barrels" must not act as a billion suffix.
    assert 300 in extract_numbers("produced 300 barrels")
    assert 300_000_000_000 not in extract_numbers("produced 300 barrels")


# ---------------------------------------------------------------------------
# contains / regex / numeric
# ---------------------------------------------------------------------------


def test_contains_case_insensitive_pass():
    case = _case(
        expected={"contains": ["Nvidia", "revenue"]},
        scoring=[{"type": "contains"}],
    )
    score = score_contains(case, _result("NVIDIA reported record REVENUE."))
    assert score.passed


def test_contains_missing_string_fails_with_detail():
    case = _case(expected={"contains": ["Nvidia"]}, scoring=[{"type": "contains"}])
    score = score_contains(case, _result("AMD reported revenue."))
    assert not score.passed
    assert "Nvidia" in score.detail


def test_regex_pass_and_fail():
    case = _case(expected={"regex": r"10-[QK]"}, scoring=[{"type": "regex"}])
    assert score_regex(case, _result("per its latest 10-Q filing")).passed
    assert not score_regex(case, _result("per its latest filing")).passed


def test_numeric_within_tolerance():
    case = _case(
        expected={"numeric": {"value": 81_615_000_000, "tolerance_pct": 2.0}},
        scoring=[{"type": "numeric"}],
    )
    assert score_numeric(case, _result("revenue was $81.6 billion")).passed


def test_numeric_outside_tolerance_fails():
    case = _case(
        expected={"numeric": {"value": 81_615_000_000, "tolerance_pct": 1.0}},
        scoring=[{"type": "numeric"}],
    )
    score = score_numeric(case, _result("revenue was $70 billion"))
    assert not score.passed


def test_numeric_ignores_unrelated_numbers():
    case = _case(
        expected={"numeric": {"value": 81_615_000_000, "tolerance_pct": 1.0}},
        scoring=[{"type": "numeric"}],
    )
    # Q1, 2026, 39% — none should accidentally pass.
    score = score_numeric(case, _result("In Q1 2026 margins were 39%"))
    assert not score.passed


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------

_TRAJ_EXPECTED = {
    "trajectory": [
        {"tool": "lookup_cik"},
        {"tool": "get_company_facts", "args_contain": {"cik": "0001045810"}},
    ]
}


def _traj_case(exact: bool = False) -> EvalCase:
    return _case(
        expected=_TRAJ_EXPECTED,
        scoring=[{"type": "trajectory", "exact": exact}],
    )


def test_trajectory_subsequence_allows_extra_calls():
    calls = [
        ToolCall("transfer_to_agent", {"agent_name": "retrieval"}),
        ToolCall("lookup_cik", {"ticker": "NVDA"}),
        ToolCall("get_recent_filings", {"cik": "0001045810"}),
        ToolCall("get_company_facts", {"cik": "0001045810"}),
    ]
    entry = _traj_case().scoring[0]
    assert score_trajectory(_traj_case(), entry, _result(tool_calls=calls)).passed


def test_trajectory_wrong_order_fails():
    calls = [
        ToolCall("get_company_facts", {"cik": "0001045810"}),
        ToolCall("lookup_cik", {"ticker": "NVDA"}),
    ]
    case = _traj_case()
    assert not score_trajectory(case, case.scoring[0], _result(tool_calls=calls)).passed


def test_trajectory_args_contain_mismatch_fails():
    calls = [
        ToolCall("lookup_cik", {"ticker": "NVDA"}),
        ToolCall("get_company_facts", {"cik": "0000320193"}),  # wrong CIK
    ]
    case = _traj_case()
    assert not score_trajectory(case, case.scoring[0], _result(tool_calls=calls)).passed


def test_trajectory_exact_mode_rejects_extra_calls():
    calls = [
        ToolCall("lookup_cik", {"ticker": "NVDA"}),
        ToolCall("get_recent_filings", {"cik": "0001045810"}),
        ToolCall("get_company_facts", {"cik": "0001045810"}),
    ]
    case = _traj_case(exact=True)
    assert not score_trajectory(case, case.scoring[0], _result(tool_calls=calls)).passed


def test_trajectory_exact_mode_pass():
    calls = [
        ToolCall("lookup_cik", {"ticker": "NVDA"}),
        ToolCall("get_company_facts", {"cik": "0001045810"}),
    ]
    case = _traj_case(exact=True)
    assert score_trajectory(case, case.scoring[0], _result(tool_calls=calls)).passed


# ---------------------------------------------------------------------------
# judge JSON parsing (no LLM calls in unit tests — parse logic only)
# ---------------------------------------------------------------------------


def test_judge_parse_plain_json():
    parsed = _parse_judge_json('{"pass": true, "reason": "looks right"}')
    assert parsed == {"pass": True, "reason": "looks right"}


def test_judge_parse_code_fenced_json():
    raw = '```json\n{"pass": false, "reason": "no citation"}\n```'
    parsed = _parse_judge_json(raw)
    assert parsed == {"pass": False, "reason": "no citation"}


def test_judge_parse_garbage_returns_none():
    assert _parse_judge_json("The answer looks fine to me!") is None


def test_judge_parse_missing_pass_key_returns_none():
    assert _parse_judge_json('{"reason": "no verdict"}') is None


def test_judge_parse_non_bool_pass_returns_none():
    assert _parse_judge_json('{"pass": "yes", "reason": "x"}') is None
