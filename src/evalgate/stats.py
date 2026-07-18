"""Statistical utilities: pass rate, Wilson score confidence interval, flakiness classification."""

from __future__ import annotations

import math

from scipy.stats import norm


def wilson_interval(passes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate.

    Wilson beats the naive p̂ ± 1.96·SE interval in exactly our regime: small N
    and extreme rates. The naive interval collapses to zero width at p̂ = 0 or 1
    (claiming certainty from 5 trials) and can exceed [0, 1]. Wilson pulls the
    centre toward 0.5 and always stays inside [0, 1].

    scipy is used only for the z-value; the formula itself is implemented here.
    """
    if trials == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = passes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_flaky(passes: int, trials: int, flaky_ci_width: float) -> bool:
    """A case is flaky when it partially passes AND the CI is too wide to trust.

    0 < pass_rate < 1 alone is not enough: 7/8 with a narrow CI is a mostly-
    reliable case, not a flaky one. The width test separates "noisy, needs a
    human look" from "consistently good/bad with a bit of noise".
    """
    if trials == 0 or passes == 0 or passes == trials:
        return False
    low, high = wilson_interval(passes, trials)
    return (high - low) > flaky_ci_width
