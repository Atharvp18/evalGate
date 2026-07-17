"""Trajectory scorer — checks that expected tool calls appear as an ordered subsequence.

Subsequence (not exact-sequence) matching is the right default for agents:
an agent may legitimately take extra steps (a retry, an extra lookup, a
transfer between sub-agents) without being wrong. What matters is that the
required calls happened in the required order. Strict mode (`exact: true`)
is available for cases where any extra call is itself a failure.
"""

from __future__ import annotations

from typing import Any

from evalgate.adapter import AgentRunResult
from evalgate.schema import EvalCase, ScoringEntry, TrajectoryStep
from evalgate.scorers.base import ScoreResult


def _step_matches(step: TrajectoryStep, call: dict[str, Any]) -> bool:
    """One expected step vs one actual tool call: name equal + args_contain subset."""
    if step.tool != call["name"]:
        return False
    args = call.get("args", {})
    # Partial dict match: every expected key present with equal stringified value.
    return all(k in args and str(args[k]) == str(v) for k, v in step.args_contain.items())


def score_trajectory(case: EvalCase, entry: ScoringEntry, result: AgentRunResult) -> ScoreResult:
    """Pass if expected.trajectory appears as an ordered subsequence of tool calls."""
    expected = case.expected.trajectory
    actual = [{"name": tc.name, "args": tc.args} for tc in result.tool_calls]
    actual_names = [c["name"] for c in actual]

    if entry.exact:
        if len(actual) == len(expected) and all(
            _step_matches(s, c) for s, c in zip(expected, actual, strict=True)
        ):
            return ScoreResult("trajectory", True, "exact match")
        return ScoreResult(
            "trajectory",
            False,
            f"exact mode: expected {[s.tool for s in expected]}, got {actual_names}",
        )

    # Subsequence scan: advance through expected steps as actual calls match.
    idx = 0
    for call in actual:
        if idx < len(expected) and _step_matches(expected[idx], call):
            idx += 1
    if idx == len(expected):
        return ScoreResult("trajectory", True, f"all {len(expected)} step(s) matched in order")
    return ScoreResult(
        "trajectory",
        False,
        f"matched {idx}/{len(expected)} step(s); "
        f"next unmatched: {expected[idx].tool!r}; actual calls: {actual_names}",
    )
