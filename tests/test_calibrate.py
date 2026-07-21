"""Unit tests for calibrate.py — sampling, CSV round-trip, kappa computation.

No LLM calls: judge verdicts here are pre-baked TrialResult fixtures, same
convention as test_store.py.
"""

from __future__ import annotations

import csv

from evalgate.calibrate import (
    LabeledRow,
    _extract_input,
    compute_calibration,
    export_for_labeling,
    load_labels,
    sample_judge_trials,
    write_calibration_report,
)
from evalgate.config import EvalGateConfig
from evalgate.schema import CaseResult, EvalCase, RunReport, TrialResult
from evalgate.scorers.judge import JUDGE_PROMPT_TEMPLATE
from evalgate.store import connect, save_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _judge_prompt(question: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(rubric="be correct", input=question, final_text="ignored")


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        description="test",
        input="q",
        expected={"contains": ["x"]},
        scoring=[{"type": "judge", "rubric": "be correct"}],
    )


def _judge_trial(trial_idx: int, question: str, answer: str, judge_pass: bool) -> TrialResult:
    return TrialResult(
        trial_idx=trial_idx,
        passed=judge_pass,
        final_text=answer,
        scores=[
            {"scorer_type": "judge", "passed": judge_pass, "detail": "reason", "extra": {}},
        ],
        judge_prompt=_judge_prompt(question),
        judge_response=f'{{"pass": {str(judge_pass).lower()}, "reason": "because"}}',
    )


def _seed_db(conn):
    """Two cases, three judge trials each, mixed pass/fail."""
    report = RunReport(
        cases=[
            CaseResult(
                case_id="case_a",
                trials=[
                    _judge_trial(0, "What was A's revenue?", "A made $1", True),
                    _judge_trial(1, "What was A's revenue?", "A made $2", False),
                    _judge_trial(2, "What was A's revenue?", "A made $1", True),
                ],
            ),
            CaseResult(
                case_id="case_b",
                trials=[
                    _judge_trial(0, "What was B's revenue?", "B made $9", True),
                    _judge_trial(1, "What was B's revenue?", "B made $9", True),
                    _judge_trial(2, "What was B's revenue?", "B made $8", False),
                ],
            ),
        ]
    )
    save_run(conn, report, [_case("case_a"), _case("case_b")], EvalGateConfig())


# ---------------------------------------------------------------------------
# _extract_input
# ---------------------------------------------------------------------------


def test_extract_input_recovers_question():
    prompt = _judge_prompt("What was Nvidia's revenue?")
    assert _extract_input(prompt) == "What was Nvidia's revenue?"


def test_extract_input_falls_back_on_missing_markers():
    assert _extract_input("not a real prompt") == "not a real prompt"


# ---------------------------------------------------------------------------
# sample_judge_trials
# ---------------------------------------------------------------------------


def test_sample_judge_trials_stratifies_across_cases():
    conn = connect(":memory:")
    _seed_db(conn)

    sampled = sample_judge_trials(conn, sample_size=4)
    assert len(sampled) == 4
    case_ids = [t.case_id for t in sampled]
    # Round-robin means both cases should be represented, not one case's 3
    # trials taking every slot before the other case gets a look-in.
    assert case_ids[0] != case_ids[1]


def test_sample_judge_trials_caps_at_available():
    conn = connect(":memory:")
    _seed_db(conn)
    sampled = sample_judge_trials(conn, sample_size=1000)
    assert len(sampled) == 6


def test_sample_judge_trials_empty_db():
    conn = connect(":memory:")
    assert sample_judge_trials(conn, sample_size=10) == []


# ---------------------------------------------------------------------------
# export_for_labeling / load_labels round-trip
# ---------------------------------------------------------------------------


def test_export_and_load_round_trip(tmp_path):
    conn = connect(":memory:")
    _seed_db(conn)

    out = tmp_path / "labels.csv"
    n = export_for_labeling(conn, out, sample_size=100)
    assert n == 6

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert rows[0]["human_label"] == ""
    assert rows[0]["input"]  # question text was recovered, not blank
    # final_text must be the agent's answer, not the judge's JSON verdict.
    assert rows[0]["final_text"].startswith(("A made", "B made"))
    assert '"pass"' not in rows[0]["final_text"]

    # No labels filled in yet -> load_labels should skip everything.
    labelled, skipped = load_labels(out)
    assert labelled == []
    assert skipped == 6

    # Fill in labels agreeing with the judge for every row.
    for row in rows:
        row["human_label"] = "1" if row["judge_pass"] == "True" else "0"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labelled, skipped = load_labels(out)
    assert skipped == 0
    assert len(labelled) == 6


# ---------------------------------------------------------------------------
# compute_calibration
# ---------------------------------------------------------------------------


def _row(trial_id: int, judge_pass: bool, human_label: bool) -> LabeledRow:
    return LabeledRow(
        trial_id=trial_id, case_id="c", judge_pass=judge_pass, human_label=human_label
    )


def test_compute_calibration_perfect_agreement():
    rows = [_row(1, True, True), _row(2, False, False), _row(3, True, True)]
    result = compute_calibration(rows)
    assert result.n == 3
    assert result.agreement_pct == 1.0
    assert result.kappa == 1.0
    assert result.disagreements == []


def test_compute_calibration_hand_computed_kappa():
    # 4 pass/pass, 2 fail/fail, 1 disagreement (judge pass, human fail).
    rows = [
        _row(1, True, True),
        _row(2, True, True),
        _row(3, True, True),
        _row(4, True, True),
        _row(5, False, False),
        _row(6, False, False),
        _row(7, True, False),
    ]
    result = compute_calibration(rows)
    assert result.n == 7
    assert result.agreement_pct == 6 / 7
    # p_o = 6/7. p_e = (marginals): human pass=4/7, judge pass=5/7 -> 20/49;
    # human fail=3/7, judge fail=2/7 -> 6/49. p_e = 26/49.
    p_o = 6 / 7
    p_e = (4 / 7) * (5 / 7) + (3 / 7) * (2 / 7)
    expected_kappa = (p_o - p_e) / (1 - p_e)
    assert abs(result.kappa - expected_kappa) < 1e-9
    assert len(result.disagreements) == 1
    assert result.disagreements[0].trial_id == 7


def test_compute_calibration_confusion_matrix_layout():
    # 1 TN, 1 FP, 1 FN, 1 TP.
    rows = [
        _row(1, False, False),  # TN
        _row(2, True, False),  # FP (human fail, judge pass)
        _row(3, False, True),  # FN (human pass, judge fail)
        _row(4, True, True),  # TP
    ]
    result = compute_calibration(rows)
    assert result.confusion == [[1, 1], [1, 1]]


def test_compute_calibration_empty():
    result = compute_calibration([])
    assert result.n == 0
    assert result.agreement_pct == 0.0


# ---------------------------------------------------------------------------
# write_calibration_report
# ---------------------------------------------------------------------------


def test_write_calibration_report(tmp_path):
    rows = [_row(1, True, True), _row(2, False, True)]
    result = compute_calibration(rows)
    out = tmp_path / "report.md"
    write_calibration_report(result, "gemini/gemini-3.1-flash-lite", out)

    text = out.read_text(encoding="utf-8")
    assert "gemini/gemini-3.1-flash-lite" in text
    assert "Cohen's kappa" in text
    assert "Disagreements (1)" in text
    assert "| 2 | c | False | True |" in text
