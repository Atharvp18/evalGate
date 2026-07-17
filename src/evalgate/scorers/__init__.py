"""Scorer dispatch — applies each ScoringEntry of a case to a trial result."""

from __future__ import annotations

from evalgate.adapter import AgentRunResult
from evalgate.config import EvalGateConfig
from evalgate.schema import EvalCase
from evalgate.scorers.base import ScoreResult
from evalgate.scorers.deterministic import score_contains, score_numeric, score_regex
from evalgate.scorers.judge import score_judge
from evalgate.scorers.trajectory import score_trajectory

__all__ = ["ScoreResult", "score_trial"]


async def score_trial(
    case: EvalCase,
    result: AgentRunResult,
    config: EvalGateConfig,
    llm_call_counter: list[int],
) -> list[ScoreResult]:
    """Run every scorer configured on the case against one trial's result.

    Judge scorers count against max_llm_calls_per_run; when the cap is hit
    the judge score fails with a clear detail instead of calling the model.
    """
    scores: list[ScoreResult] = []
    for entry in case.scoring:
        if entry.type == "contains":
            scores.append(score_contains(case, result))
        elif entry.type == "regex":
            scores.append(score_regex(case, result))
        elif entry.type == "numeric":
            scores.append(score_numeric(case, result))
        elif entry.type == "trajectory":
            scores.append(score_trajectory(case, entry, result))
        elif entry.type == "judge":
            llm_call_counter[0] += 1
            if llm_call_counter[0] > config.max_llm_calls_per_run:
                scores.append(
                    ScoreResult(
                        "judge",
                        False,
                        f"max_llm_calls_exceeded (limit={config.max_llm_calls_per_run})",
                    )
                )
            else:
                scores.append(await score_judge(case, entry, result, config.judge_model))
    return scores
