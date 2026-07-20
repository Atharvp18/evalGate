"""Unit tests for compare.py — the two-condition regression gate."""

from __future__ import annotations

from evalgate.compare import compare_runs, has_regressions
from evalgate.stats import wilson_interval

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(case_id: str, passes: int, trials: int) -> dict:
    """Build a case_results-shaped row with a real Wilson CI."""
    ci_low, ci_high = wilson_interval(passes, trials)
    return {
        "case_id": case_id,
        "pass_rate": passes / trials,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _verdict(baseline: dict, current: dict, margin: float = 0.10) -> str:
    (comparison,) = compare_runs([baseline], [current], margin)
    return comparison.verdict


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clear_regression_flagged():
    # 8/8 → 1/8: huge drop, and the 1/8 CI (≈[0.02, 0.47]) excludes 1.0.
    assert _verdict(_row("a", 8, 8), _row("a", 1, 8)) == "regressed"


def test_drop_within_margin_is_ok():
    # 8/8 → 15/16 = 0.9375: drop of 0.0625 < margin 0.10 → condition 1 fails.
    assert _verdict(_row("a", 8, 8), _row("a", 15, 16)) == "ok"


def test_drop_past_margin_but_ci_contains_baseline_is_ok():
    # 6/8 = 0.75 → 5/8 = 0.625: drop 0.125 > margin, but the 5/8 CI
    # (≈[0.31, 0.86]) still contains 0.75 — could be sampling noise → ok.
    base, cur = _row("a", 6, 8), _row("a", 5, 8)
    assert cur["ci_low"] <= base["pass_rate"] <= cur["ci_high"]  # test the premise
    assert _verdict(base, cur) == "ok"


def test_improvement_flagged_informationally():
    assert _verdict(_row("a", 4, 8), _row("a", 8, 8)) == "improved"


def test_new_case_never_gates():
    (c,) = compare_runs([], [_row("brand_new", 0, 8)], 0.10)
    assert c.verdict == "new"
    assert c.baseline_rate is None
    assert not has_regressions([c])


def test_removed_case_warns_not_gates():
    (c,) = compare_runs([_row("gone", 8, 8)], [], 0.10)
    assert c.verdict == "removed"
    assert c.current_rate is None
    assert not has_regressions([c])


def test_has_regressions_mixed():
    comparisons = compare_runs(
        [_row("stable", 8, 8), _row("broken", 8, 8)],
        [_row("stable", 8, 8), _row("broken", 0, 8)],
        0.10,
    )
    assert has_regressions(comparisons)
    verdicts = {c.case_id: c.verdict for c in comparisons}
    assert verdicts == {"stable": "ok", "broken": "regressed"}


def test_output_sorted_by_case_id():
    comparisons = compare_runs(
        [_row("zeta", 8, 8)], [_row("zeta", 8, 8), _row("alpha", 8, 8)], 0.10
    )
    assert [c.case_id for c in comparisons] == ["alpha", "zeta"]
