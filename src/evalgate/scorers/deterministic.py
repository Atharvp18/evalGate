"""Deterministic scorers: contains, regex, and numeric-tolerance."""

from __future__ import annotations

import re

from evalgate.adapter import AgentRunResult
from evalgate.schema import EvalCase
from evalgate.scorers.base import ScoreResult

# Scale words the number normalizer understands, mapped to multipliers.
_SCALES: dict[str, float] = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "t": 1e12,
}

# A number like 60,922 / 60922000000 / 60.92, optionally preceded by $ and
# followed by a scale word ("billion", "B", "million", ...).
_NUMBER_RE = re.compile(
    r"""
    \$?\s*
    (-?\d{1,3}(?:,\d{3})+(?:\.\d+)?    # comma-grouped: 60,922 or 1,234.5
     |-?\d+(?:\.\d+)?)                 # or plain: 60922000000 / 60.92
    \s*
    (thousand|million|billion|trillion|bn|mn|k|m|b|t)?\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_numbers(text: str) -> list[float]:
    """Extract candidate numeric values from text, normalising scale words.

    "$60.9 billion" → 60_900_000_000; "60,922 million" → 60_922_000_000;
    "60922000000" → 60_922_000_000. Numbers immediately followed by '%' are
    skipped — percentages are almost never the dollar answer being checked.
    """
    candidates: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        # Skip percentages: "grew 12%" must not produce candidate 12.
        rest = text[m.end() :].lstrip()
        if rest.startswith("%"):
            continue
        value = float(m.group(1).replace(",", ""))
        scale = m.group(2)
        if scale:
            value *= _SCALES[scale.lower()]
        candidates.append(value)
    return candidates


def score_contains(case: EvalCase, result: AgentRunResult) -> ScoreResult:
    """Pass if every expected string appears in final_text (case-insensitive)."""
    text = result.final_text.lower()
    missing = [s for s in case.expected.contains if s.lower() not in text]
    if missing:
        return ScoreResult("contains", False, f"missing: {missing}")
    return ScoreResult("contains", True, f"all {len(case.expected.contains)} string(s) found")


def score_regex(case: EvalCase, result: AgentRunResult) -> ScoreResult:
    """Pass if expected.regex matches anywhere in final_text."""
    pattern = case.expected.regex
    assert pattern is not None  # schema validator guarantees this
    if re.search(pattern, result.final_text):
        return ScoreResult("regex", True, f"pattern {pattern!r} matched")
    return ScoreResult("regex", False, f"pattern {pattern!r} not found")


def score_numeric(case: EvalCase, result: AgentRunResult) -> ScoreResult:
    """Pass if any number extracted from final_text is within tolerance of expected."""
    expected = case.expected.numeric
    assert expected is not None  # schema validator guarantees this
    candidates = extract_numbers(result.final_text)
    tolerance = abs(expected.value) * expected.tolerance_pct / 100
    for c in candidates:
        if abs(c - expected.value) <= tolerance:
            return ScoreResult(
                "numeric",
                True,
                f"found {c:g} within {expected.tolerance_pct}% of {expected.value:g}",
            )
    return ScoreResult(
        "numeric",
        False,
        f"no candidate within {expected.tolerance_pct}% of {expected.value:g} "
        f"(candidates: {candidates[:10]})",
    )
