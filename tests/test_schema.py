"""Unit tests for schema.py — EvalCase validation and result dataclasses."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalgate.schema import (
    CaseResult,
    EvalCase,
    NumericExpected,
    RunReport,
    TrialResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_judge_case(**overrides) -> dict:
    """Minimal valid EvalCase dict with a judge scorer."""
    base = {
        "id": "test_case",
        "description": "A test case",
        "input": "What is X?",
        "scoring": [{"type": "judge", "rubric": "The answer is correct."}],
    }
    base.update(overrides)
    return base


def _valid_numeric_case(**overrides) -> dict:
    base = {
        "id": "numeric_case",
        "description": "A numeric case",
        "input": "What is X?",
        "expected": {"numeric": {"value": 100.0, "tolerance_pct": 5.0}},
        "scoring": [{"type": "numeric"}],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Valid cases
# ---------------------------------------------------------------------------


class TestValidCase:
    def test_minimal_judge_case(self) -> None:
        case = EvalCase(**_valid_judge_case())
        assert case.id == "test_case"
        assert case.trials == 6  # default
        assert case.pass_policy == "all"
        assert case.thresholds.pass_rate_min == 0.75

    def test_minimal_numeric_case(self) -> None:
        case = EvalCase(**_valid_numeric_case())
        assert case.expected.numeric is not None
        assert case.expected.numeric.value == 100.0

    def test_full_case_all_fields(self) -> None:
        data = {
            "id": "full_case",
            "description": "desc",
            "input": "query",
            "tags": ["retrieval", "numeric"],
            "trials": 8,
            "expected": {
                "numeric": {"value": 81615000000, "tolerance_pct": 2.0},
                "contains": ["Nvidia"],
                "trajectory": [
                    {"tool": "lookup_cik"},
                    {"tool": "get_company_facts", "args_contain": {"cik": "0001045810"}},
                ],
            },
            "scoring": [
                {"type": "numeric"},
                {"type": "trajectory"},
                {"type": "judge", "rubric": "States revenue correctly."},
            ],
            "pass_policy": "all",
            "thresholds": {"pass_rate_min": 0.8},
        }
        case = EvalCase(**data)
        assert case.trials == 8
        assert len(case.expected.trajectory) == 2
        assert case.expected.trajectory[1].args_contain == {"cik": "0001045810"}
        assert case.thresholds.pass_rate_min == 0.8

    def test_pass_policy_any(self) -> None:
        data = _valid_judge_case(pass_policy="any")
        case = EvalCase(**data)
        assert case.pass_policy == "any"

    def test_contains_scorer_with_contains_list(self) -> None:
        data = {
            "id": "contains_case",
            "description": "d",
            "input": "q",
            "expected": {"contains": ["foo", "bar"]},
            "scoring": [{"type": "contains"}],
        }
        case = EvalCase(**data)
        assert case.expected.contains == ["foo", "bar"]


# ---------------------------------------------------------------------------
# ID validation
# ---------------------------------------------------------------------------


class TestIdValidation:
    @pytest.mark.parametrize("valid_id", ["abc", "a_b_c", "a1", "abc123", "nvda_revenue_q1"])
    def test_valid_ids(self, valid_id: str) -> None:
        case = EvalCase(**_valid_judge_case(id=valid_id))
        assert case.id == valid_id

    @pytest.mark.parametrize("bad_id", ["CamelCase", "has space", "has-dash", "HAS.DOT", ""])
    def test_invalid_ids_raise(self, bad_id: str) -> None:
        with pytest.raises(ValidationError, match="id"):
            EvalCase(**_valid_judge_case(id=bad_id))


# ---------------------------------------------------------------------------
# Trials validation
# ---------------------------------------------------------------------------


class TestTrialsValidation:
    @pytest.mark.parametrize("n", [1, 6, 20])
    def test_valid_trial_counts(self, n: int) -> None:
        case = EvalCase(**_valid_judge_case(trials=n))
        assert case.trials == n

    @pytest.mark.parametrize("n", [0, 21, -1, 100])
    def test_invalid_trial_counts(self, n: int) -> None:
        with pytest.raises(ValidationError, match="trials"):
            EvalCase(**_valid_judge_case(trials=n))


# ---------------------------------------------------------------------------
# Scoring / expected cross-validation
# ---------------------------------------------------------------------------


class TestScoringValidation:
    def test_numeric_scorer_without_numeric_expected_raises(self) -> None:
        data = {
            "id": "bad",
            "description": "d",
            "input": "q",
            "scoring": [{"type": "numeric"}],
        }
        with pytest.raises(ValidationError, match="numeric"):
            EvalCase(**data)

    def test_trajectory_scorer_without_trajectory_expected_raises(self) -> None:
        data = {
            "id": "bad",
            "description": "d",
            "input": "q",
            "scoring": [{"type": "trajectory"}],
        }
        with pytest.raises(ValidationError, match="trajectory"):
            EvalCase(**data)

    def test_judge_scorer_without_rubric_raises(self) -> None:
        data = {
            "id": "bad",
            "description": "d",
            "input": "q",
            "scoring": [{"type": "judge"}],
        }
        with pytest.raises(ValidationError, match="rubric"):
            EvalCase(**data)

    def test_regex_scorer_without_regex_expected_raises(self) -> None:
        data = {
            "id": "bad",
            "description": "d",
            "input": "q",
            "scoring": [{"type": "regex"}],
        }
        with pytest.raises(ValidationError, match="regex"):
            EvalCase(**data)

    def test_contains_scorer_without_contains_list_raises(self) -> None:
        data = {
            "id": "bad",
            "description": "d",
            "input": "q",
            "scoring": [{"type": "contains"}],
        }
        with pytest.raises(ValidationError, match="contains"):
            EvalCase(**data)

    def test_empty_scoring_list_raises(self) -> None:
        data = {
            "id": "bad",
            "description": "d",
            "input": "q",
            "scoring": [],
        }
        with pytest.raises(ValidationError, match="scoring"):
            EvalCase(**data)


# ---------------------------------------------------------------------------
# Extra fields rejected
# ---------------------------------------------------------------------------


class TestExtraFieldsRejected:
    def test_extra_field_on_case_raises(self) -> None:
        data = _valid_judge_case(unknown_key="surprise")
        with pytest.raises(ValidationError, match="unknown_key"):
            EvalCase(**data)

    def test_extra_field_on_expected_raises(self) -> None:
        data = {
            "id": "bad",
            "description": "d",
            "input": "q",
            "expected": {"not_a_real_field": 42},
            "scoring": [{"type": "judge", "rubric": "ok"}],
        }
        with pytest.raises(ValidationError, match="not_a_real_field"):
            EvalCase(**data)

    def test_extra_field_on_numeric_expected_raises(self) -> None:
        data = {
            "id": "bad",
            "description": "d",
            "input": "q",
            "expected": {"numeric": {"value": 100.0, "tolerance_pct": 1.0, "extra": "x"}},
            "scoring": [{"type": "numeric"}],
        }
        with pytest.raises(ValidationError, match="extra"):
            EvalCase(**data)


# ---------------------------------------------------------------------------
# NumericExpected
# ---------------------------------------------------------------------------


class TestNumericExpected:
    def test_positive_tolerance_required(self) -> None:
        with pytest.raises(ValidationError, match="tolerance_pct"):
            NumericExpected(value=100.0, tolerance_pct=0.0)

    def test_negative_tolerance_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tolerance_pct"):
            NumericExpected(value=100.0, tolerance_pct=-1.0)

    def test_default_tolerance(self) -> None:
        n = NumericExpected(value=50.0)
        assert n.tolerance_pct == 1.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    def _make_trial(self, passed: bool, latency: float = 100.0, tokens_in: int = 0) -> TrialResult:
        return TrialResult(
            trial_idx=0,
            passed=passed,
            latency_ms=latency,
            input_tokens=tokens_in,
        )

    def test_case_result_pass_rate(self) -> None:
        cr = CaseResult(
            case_id="x",
            trials=[self._make_trial(True), self._make_trial(False), self._make_trial(True)],
        )
        assert cr.passes == 2
        assert cr.pass_rate == pytest.approx(2 / 3)

    def test_case_result_p50_latency(self) -> None:
        cr = CaseResult(
            case_id="x",
            trials=[
                self._make_trial(True, latency=100.0),
                self._make_trial(True, latency=200.0),
                self._make_trial(True, latency=300.0),
            ],
        )
        assert cr.p50_latency_ms == pytest.approx(200.0)

    def test_case_result_zero_trials(self) -> None:
        cr = CaseResult(case_id="x", trials=[])
        assert cr.pass_rate == 0.0
        assert cr.latencies_ms == []

    def test_run_report_totals(self) -> None:
        cr1 = CaseResult(
            case_id="a",
            trials=[TrialResult(trial_idx=0, passed=True, input_tokens=100, output_tokens=50)],
        )
        cr2 = CaseResult(
            case_id="b",
            trials=[TrialResult(trial_idx=0, passed=False, input_tokens=200, output_tokens=80)],
        )
        rpt = RunReport(cases=[cr1, cr2])
        assert rpt.total_input_tokens == 300
        assert rpt.total_output_tokens == 130
