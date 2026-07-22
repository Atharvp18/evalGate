"""Unit tests for study/run_study.py's pure logic — naive-mode case filtering,
report-shape conversion, and the naive catch rule. No LLM calls: RunReport
objects are built directly from TrialResult fixtures, same convention as
test_store.py and test_compare.py.
"""

from __future__ import annotations

import asyncio

import pytest
from study.injections import INJECTIONS
from study.run_study import _build_adapter, case_result_rows, naive_cases, naive_regressions

from evalgate.config import EvalGateConfig
from evalgate.schema import CaseResult, EvalCase, RunReport, TrialResult


def _case(case_id: str, scoring: list[dict]) -> EvalCase:
    return EvalCase(
        id=case_id,
        description="test",
        input="q",
        expected={"contains": ["x"], "trajectory": [{"tool": "lookup_cik"}]},
        scoring=scoring,
        trials=4,
    )


def _trial(idx: int, passed: bool) -> TrialResult:
    return TrialResult(trial_idx=idx, passed=passed)


# ---------------------------------------------------------------------------
# naive_cases
# ---------------------------------------------------------------------------


def test_naive_cases_keeps_only_deterministic_scorers():
    case = _case(
        "c1",
        [{"type": "contains"}, {"type": "trajectory"}, {"type": "judge", "rubric": "r"}],
    )
    (naive,) = naive_cases([case])
    assert [e.type for e in naive.scoring] == ["contains"]


def test_naive_cases_forces_one_trial():
    case = _case("c1", [{"type": "contains"}])
    case.trials = 4
    (naive,) = naive_cases([case])
    assert naive.trials == 1


def test_naive_cases_does_not_mutate_original():
    case = _case("c1", [{"type": "contains"}, {"type": "judge", "rubric": "r"}])
    naive_cases([case])
    assert len(case.scoring) == 2
    assert case.trials == 4


def test_naive_cases_raises_when_no_deterministic_scorer():
    case = _case("c1", [{"type": "judge", "rubric": "r"}])
    with pytest.raises(ValueError, match="c1"):
        naive_cases([case])


# ---------------------------------------------------------------------------
# case_result_rows
# ---------------------------------------------------------------------------


def test_case_result_rows_computes_wilson_ci():
    report = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, True), _trial(1, False)])])
    (row,) = case_result_rows(report)
    assert row["case_id"] == "c1"
    assert row["pass_rate"] == 0.5
    assert 0.0 <= row["ci_low"] < 0.5 < row["ci_high"] <= 1.0


# ---------------------------------------------------------------------------
# naive_regressions
# ---------------------------------------------------------------------------


def test_naive_regressions_flags_pass_to_fail_flip():
    before = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, True)])])
    after = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, False)])])
    assert naive_regressions(before, after) == ["c1"]


def test_naive_regressions_ignores_fail_to_fail():
    before = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, False)])])
    after = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, False)])])
    assert naive_regressions(before, after) == []


def test_naive_regressions_ignores_pass_to_pass():
    before = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, True)])])
    after = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, True)])])
    assert naive_regressions(before, after) == []


def test_naive_regressions_ignores_improvement():
    before = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, False)])])
    after = RunReport(cases=[CaseResult(case_id="c1", trials=[_trial(0, True)])])
    assert naive_regressions(before, after) == []


# ---------------------------------------------------------------------------
# _build_adapter — no LLM calls, just constructs the ADK agent object graph.
# ---------------------------------------------------------------------------


def test_build_adapter_lets_temperature_injection_override_config_default():
    # Regression test: build_agent(temperature=cfg.temperature, **build_kwargs)
    # raised "got multiple values for keyword argument 'temperature'" whenever
    # an injection's build_kwargs also set "temperature" (injection #5).
    cfg = EvalGateConfig()
    injection = next(i for i in INJECTIONS if i.id == "temperature_1_0")
    with injection.apply():
        adapter = asyncio.run(_build_adapter(injection.build_kwargs, cfg))
    assert adapter is not None


def test_build_adapter_works_for_every_injection():
    cfg = EvalGateConfig()
    for injection in INJECTIONS:
        with injection.apply():
            adapter = asyncio.run(_build_adapter(injection.build_kwargs, cfg))
        assert adapter is not None


def test_naive_regressions_multiple_cases():
    before = RunReport(
        cases=[
            CaseResult(case_id="c1", trials=[_trial(0, True)]),
            CaseResult(case_id="c2", trials=[_trial(0, True)]),
            CaseResult(case_id="c3", trials=[_trial(0, False)]),
        ]
    )
    after = RunReport(
        cases=[
            CaseResult(case_id="c1", trials=[_trial(0, False)]),
            CaseResult(case_id="c2", trials=[_trial(0, True)]),
            CaseResult(case_id="c3", trials=[_trial(0, True)]),
        ]
    )
    assert naive_regressions(before, after) == ["c1"]
