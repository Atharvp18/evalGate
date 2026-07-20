"""Baseline comparison logic — two-condition regression gate with exit codes for CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["ok", "regressed", "improved", "new", "removed"]


@dataclass
class CaseComparison:
    """Verdict for one case_id across baseline and current runs."""

    case_id: str
    verdict: Verdict
    baseline_rate: float | None  # None when the case is new
    current_rate: float | None  # None when the case was removed
    current_ci: tuple[float, float] | None  # None when the case was removed

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "verdict": self.verdict,
            "baseline_rate": self.baseline_rate,
            "current_rate": self.current_rate,
            "current_ci": list(self.current_ci) if self.current_ci else None,
        }


def compare_runs(
    baseline_cases: list[dict],
    current_cases: list[dict],
    regression_margin: float,
) -> list[CaseComparison]:
    """Compare two runs' case_results rows and assign a verdict per case.

    A case REGRESSES only when BOTH conditions hold:
      1. current.pass_rate < baseline.pass_rate - regression_margin
         (the drop is big enough to care about), AND
      2. the current run's Wilson CI does not contain the baseline pass rate
         (the drop is statistically distinguishable from sampling noise).
    Condition 2 is the guard against flagging noise: with few trials the CI is
    wide, so a wobble like 6/8 → 5/8 keeps the baseline rate inside the CI and
    stays "ok". Only drops that are both large AND clear regress.

    New cases (no baseline row) are informational; removed cases warn.
    """
    baseline_by_id = {row["case_id"]: row for row in baseline_cases}
    current_by_id = {row["case_id"]: row for row in current_cases}

    comparisons: list[CaseComparison] = []
    for case_id in sorted(baseline_by_id.keys() | current_by_id.keys()):
        base = baseline_by_id.get(case_id)
        cur = current_by_id.get(case_id)

        if base is None:
            assert cur is not None
            comparisons.append(
                CaseComparison(
                    case_id, "new", None, cur["pass_rate"], (cur["ci_low"], cur["ci_high"])
                )
            )
            continue
        if cur is None:
            comparisons.append(CaseComparison(case_id, "removed", base["pass_rate"], None, None))
            continue

        ci = (cur["ci_low"], cur["ci_high"])
        dropped_past_margin = cur["pass_rate"] < base["pass_rate"] - regression_margin
        ci_excludes_baseline = not (ci[0] <= base["pass_rate"] <= ci[1])

        if dropped_past_margin and ci_excludes_baseline:
            verdict: Verdict = "regressed"
        elif cur["pass_rate"] > base["pass_rate"] + regression_margin:
            verdict = "improved"
        else:
            verdict = "ok"
        comparisons.append(
            CaseComparison(case_id, verdict, base["pass_rate"], cur["pass_rate"], ci)
        )

    return comparisons


def has_regressions(comparisons: list[CaseComparison]) -> bool:
    """True when at least one case regressed — maps to CI exit code 1."""
    return any(c.verdict == "regressed" for c in comparisons)
