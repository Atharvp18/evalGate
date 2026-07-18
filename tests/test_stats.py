"""Unit tests for stats.py — Wilson interval against hand-checked values, flakiness."""

from __future__ import annotations

import pytest

from evalgate.stats import is_flaky, wilson_interval

# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_8_of_10_matches_hand_computation():
    # Hand-computed: p̂=0.8, n=10, z=1.95996 → CI ≈ (0.4902, 0.9433).
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=0.001)
    assert high == pytest.approx(0.9433, abs=0.001)


def test_wilson_all_passes_has_nonzero_width():
    # The naive interval at 5/5 would be (1.0, 1.0) — zero width, false
    # certainty. Wilson correctly admits uncertainty at small N.
    low, high = wilson_interval(5, 5)
    assert high == 1.0
    assert low < 0.6  # hand-computed lower bound ≈ 0.566


def test_wilson_zero_passes():
    low, high = wilson_interval(0, 5)
    assert low == 0.0
    assert high > 0.4  # hand-computed upper bound ≈ 0.434


def test_wilson_stays_inside_unit_interval():
    for passes in range(0, 9):
        low, high = wilson_interval(passes, 8)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_zero_trials_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Flakiness
# ---------------------------------------------------------------------------


def test_all_pass_is_not_flaky():
    assert not is_flaky(8, 8, flaky_ci_width=0.5)


def test_all_fail_is_not_flaky():
    assert not is_flaky(0, 8, flaky_ci_width=0.5)


def test_half_passes_at_small_n_is_flaky():
    # 2/4 → Wilson CI ≈ (0.15, 0.85), width 0.70 > 0.5.
    assert is_flaky(2, 4, flaky_ci_width=0.5)


def test_partial_pass_with_narrow_ci_is_not_flaky():
    # 15/20 → CI ≈ (0.53, 0.89), width ≈ 0.36 < 0.5: mostly reliable, not flaky.
    assert not is_flaky(15, 20, flaky_ci_width=0.5)
