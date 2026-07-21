"""Judge calibration — exports trials for human labelling, computes Cohen's kappa.

The judge's prompt (JUDGE_PROMPT_TEMPLATE in scorers/judge.py) already embeds the
question between fixed markers, so the original input can be recovered straight
from the stored `judge_prompt` text. No schema change (e.g. a dedicated `input`
column) is needed just to round-trip a string we already persisted verbatim.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

_QUESTION_START = "Question given to the agent:\n"
_QUESTION_END = "\n\nAgent's final answer:"

CSV_FIELDS = ["trial_id", "case_id", "input", "final_text", "judge_pass", "human_label"]


def _extract_input(judge_prompt: str) -> str:
    """Pull the original question out of a stored judge prompt.

    Falls back to the whole prompt if the markers are ever missing (e.g. the
    prompt template changed) rather than raising — labelling can still proceed
    with slightly noisier context.
    """
    try:
        start = judge_prompt.index(_QUESTION_START) + len(_QUESTION_START)
        end = judge_prompt.index(_QUESTION_END, start)
        return judge_prompt[start:end].strip()
    except ValueError:
        return judge_prompt.strip()


@dataclass
class JudgeTrial:
    trial_id: int
    case_id: str
    input: str
    final_text: str
    judge_pass: bool


def sample_judge_trials(
    conn: sqlite3.Connection, sample_size: int = 100, run_id: int | None = None
) -> list[JudgeTrial]:
    """Fetch judge-scored trials, stratified (round-robin) across case_id.

    Round-robin across cases means a case with many trials cannot crowd out a
    case with few — every case gets a fair shot at appearing in the sample
    before any case gets a second pick.
    """
    query = (
        "SELECT t.id AS trial_id, cr.case_id, t.judge_prompt, t.final_text, t.scores_json "
        "FROM trials t JOIN case_results cr ON t.case_result_id = cr.id "
        "WHERE t.judge_prompt IS NOT NULL"
    )
    params: tuple = ()
    if run_id is not None:
        query += " AND cr.run_id = ?"
        params = (run_id,)
    query += " ORDER BY cr.case_id, t.id"

    rows = conn.execute(query, params).fetchall()

    import json

    by_case: dict[str, list[JudgeTrial]] = {}
    for row in rows:
        scores = json.loads(row["scores_json"])
        judge_score = next((s for s in scores if s["scorer_type"] == "judge"), None)
        if judge_score is None:
            continue
        trial = JudgeTrial(
            trial_id=row["trial_id"],
            case_id=row["case_id"],
            input=_extract_input(row["judge_prompt"]),
            final_text=row["final_text"],
            judge_pass=bool(judge_score["passed"]),
        )
        by_case.setdefault(row["case_id"], []).append(trial)

    sampled: list[JudgeTrial] = []
    case_queues = list(by_case.values())
    idx = 0
    while len(sampled) < sample_size and case_queues:
        queue = case_queues[idx % len(case_queues)]
        if queue:
            sampled.append(queue.pop(0))
        idx += 1
        case_queues = [q for q in case_queues if q]

    return sampled


def export_for_labeling(
    conn: sqlite3.Connection, out_path: str | Path, sample_size: int = 100
) -> int:
    """Write an unlabelled CSV of sampled judge trials. Returns the row count."""
    trials = sample_judge_trials(conn, sample_size)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for t in trials:
            writer.writerow(
                {
                    "trial_id": t.trial_id,
                    "case_id": t.case_id,
                    "input": t.input,
                    "final_text": t.final_text,
                    "judge_pass": t.judge_pass,
                    "human_label": "",
                }
            )
    return len(trials)


@dataclass
class LabeledRow:
    trial_id: int
    case_id: str
    judge_pass: bool
    human_label: bool


def load_labels(path: str | Path) -> tuple[list[LabeledRow], int]:
    """Read a labelled CSV. Returns (labelled rows, count of still-blank rows skipped)."""
    rows: list[LabeledRow] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("human_label") or "").strip()
            if label == "":
                skipped += 1
                continue
            rows.append(
                LabeledRow(
                    trial_id=int(row["trial_id"]),
                    case_id=row["case_id"],
                    judge_pass=row["judge_pass"].strip().lower() == "true",
                    human_label=label in ("1", "true", "True"),
                )
            )
    return rows, skipped


@dataclass
class CalibrationResult:
    n: int
    agreement_pct: float
    kappa: float
    confusion: list[list[int]]  # [[TN, FP], [FN, TP]] — rows=human, cols=judge
    disagreements: list[LabeledRow] = field(default_factory=list)


def compute_calibration(rows: list[LabeledRow]) -> CalibrationResult:
    """Agreement %, Cohen's kappa, and confusion matrix between judge and human labels.

    Raw agreement % overstates reliability when one class dominates (e.g. a
    judge that always says "pass" agrees with a mostly-passing human 90% of
    the time by luck alone). Kappa corrects for that chance agreement, so it
    is the number that actually says whether the judge is a good judge.
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    if not rows:
        return CalibrationResult(n=0, agreement_pct=0.0, kappa=0.0, confusion=[[0, 0], [0, 0]])

    human = [int(r.human_label) for r in rows]
    judge = [int(r.judge_pass) for r in rows]

    agree = sum(1 for h, j in zip(human, judge, strict=True) if h == j)
    agreement_pct = agree / len(rows)
    kappa = cohen_kappa_score(human, judge) if len(set(human)) > 1 or len(set(judge)) > 1 else 1.0
    confusion = confusion_matrix(human, judge, labels=[0, 1]).tolist()
    disagreements = [r for r in rows if r.human_label != r.judge_pass]

    return CalibrationResult(
        n=len(rows),
        agreement_pct=agreement_pct,
        kappa=kappa,
        confusion=confusion,
        disagreements=disagreements,
    )


def write_calibration_report(
    result: CalibrationResult, judge_model: str, out_path: str | Path = "calibration_report.md"
) -> None:
    """Render the calibration numbers as a small markdown report."""
    tn, fp = result.confusion[0]
    fn, tp = result.confusion[1]

    lines = [
        "# Judge Calibration Report",
        "",
        f"Judge model: `{judge_model}`",
        f"Labelled trials: {result.n}",
        "",
        f"- **Agreement:** {result.agreement_pct:.1%}",
        f"- **Cohen's kappa:** {result.kappa:.3f}",
        "",
        "## Confusion matrix",
        "",
        "|              | Judge: fail | Judge: pass |",
        "|--------------|-------------|-------------|",
        f"| **Human: fail** | {tn} | {fp} |",
        f"| **Human: pass** | {fn} | {tp} |",
        "",
        f"## Disagreements ({len(result.disagreements)})",
        "",
    ]
    if result.disagreements:
        lines.append("| trial_id | case_id | judge | human |")
        lines.append("|----------|---------|-------|-------|")
        for d in result.disagreements:
            lines.append(f"| {d.trial_id} | {d.case_id} | {d.judge_pass} | {d.human_label} |")
    else:
        lines.append("None — judge and human agreed on every labelled trial.")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
