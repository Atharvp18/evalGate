"""LLM-as-judge scorer — calls a configurable model via litellm with a rubric prompt.

The judge never sees the expected answer for contains/numeric cases — it
judges against the rubric only. This keeps the judge an independent signal
rather than a noisy re-implementation of the deterministic scorers, and it
is what makes calibration (Phase 7) meaningful.
"""

from __future__ import annotations

import json
import logging

from evalgate.adapter import AgentRunResult
from evalgate.schema import EvalCase, ScoringEntry
from evalgate.scorers.base import ScoreResult

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating an AI agent's answer against a rubric.

Rubric:
{rubric}

Question given to the agent:
{input}

Agent's final answer:
{final_text}

Does the answer satisfy the rubric? Reply with ONLY strict JSON, no other text:
{{"pass": true or false, "reason": "one short sentence"}}"""


def _parse_judge_json(raw: str) -> dict | None:
    """Parse the judge reply, tolerating markdown code fences. None if unparseable."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences.
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("pass"), bool):
        return parsed
    return None


async def score_judge(
    case: EvalCase,
    entry: ScoringEntry,
    result: AgentRunResult,
    model: str,
) -> ScoreResult:
    """Ask the judge model whether the answer satisfies the rubric.

    On unparseable output, retry once; still unparseable → failed with
    detail="judge_output_unparseable". The prompt and raw response are stored
    in extra for calibration (Phase 7).
    """
    import litellm  # imported lazily: slow import, only needed when a judge runs

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        rubric=entry.rubric, input=case.input, final_text=result.final_text
    )

    raw = ""
    for attempt in range(2):
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_judge_json(raw)
        if parsed is not None:
            return ScoreResult(
                "judge",
                bool(parsed["pass"]),
                str(parsed.get("reason", "")),
                extra={"judge_prompt": prompt, "judge_response": raw},
            )
        logger.warning(
            "Judge output unparseable for case %r (attempt %d): %r",
            case.id,
            attempt + 1,
            raw[:200],
        )

    return ScoreResult(
        "judge",
        False,
        "judge_output_unparseable",
        extra={"judge_prompt": prompt, "judge_response": raw},
    )
