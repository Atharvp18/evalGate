"""Async N-trial executor — runs each eval case N times with bounded concurrency."""

from __future__ import annotations

import asyncio
import logging
import traceback

from evalgate.adapter import AgentAdapter, AgentRunResult
from evalgate.config import EvalGateConfig
from evalgate.schema import CaseResult, EvalCase, RunReport, TrialResult
from evalgate.scorers import score_trial

logger = logging.getLogger(__name__)


async def _run_single_trial(
    case: EvalCase,
    trial_idx: int,
    adapter: AgentAdapter,
    config: EvalGateConfig,
    semaphore: asyncio.Semaphore,
    llm_call_counter: list[int],
    cost_per_input_token: float,
    cost_per_output_token: float,
) -> TrialResult:
    """Run one trial: acquire semaphore → adapter.run → run all configured scorers.

    The semaphore limits total concurrent adapter calls across all cases and
    trials. This serves two purposes: (1) EDGAR politeness — replay mode is
    fast but live mode hits the real API; (2) Gemini free-tier rate limits —
    too many parallel requests get 429s.

    Any exception is caught and stored as a failed trial. The run never
    crashes because one trial throws.
    """
    async with semaphore:
        # Cost guardrail: count this trial as one LLM call (the agent itself
        # triggers multiple sub-agent LLM calls internally, but we track at
        # the adapter.run() level; judge calls are counted inside score_trial).
        llm_call_counter[0] += 1
        if llm_call_counter[0] > config.max_llm_calls_per_run:
            logger.error(
                "Aborting trial %d for case %r: max_llm_calls_per_run (%d) exceeded",
                trial_idx,
                case.id,
                config.max_llm_calls_per_run,
            )
            return TrialResult(
                trial_idx=trial_idx,
                passed=False,
                failure_reason=f"max_llm_calls_exceeded (limit={config.max_llm_calls_per_run})",
            )

        try:
            result: AgentRunResult = await asyncio.wait_for(
                adapter.run(case.input),
                timeout=float(config.trial_timeout_s),
            )
        except TimeoutError:
            logger.warning(
                "Trial %d for case %r timed out after %ds",
                trial_idx,
                case.id,
                config.trial_timeout_s,
            )
            return TrialResult(
                trial_idx=trial_idx,
                passed=False,
                failure_reason=f"timeout after {config.trial_timeout_s}s",
            )
        except Exception:
            tb = traceback.format_exc()
            logger.error(
                "Trial %d for case %r raised an exception:\n%s",
                trial_idx,
                case.id,
                tb,
            )
            return TrialResult(
                trial_idx=trial_idx,
                passed=False,
                failure_reason=tb.strip(),
            )

        cost = (
            result.input_tokens * cost_per_input_token
            + result.output_tokens * cost_per_output_token
        )

        # Score the trial. Scoring failures (e.g. judge network errors) are a
        # failed trial, never a crashed run — same contract as adapter errors.
        try:
            scores = await score_trial(case, result, config, llm_call_counter)
        except Exception:
            tb = traceback.format_exc()
            logger.error("Scoring trial %d for case %r failed:\n%s", trial_idx, case.id, tb)
            return TrialResult(
                trial_idx=trial_idx,
                passed=False,
                failure_reason=f"scoring error: {tb.strip()}",
                final_text=result.final_text,
                tool_calls=[{"name": tc.name, "args": tc.args} for tc in result.tool_calls],
                raw_events=result.raw_events,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=cost,
            )

        if case.pass_policy == "all":
            passed = all(s.passed for s in scores)
        else:
            passed = any(s.passed for s in scores)
        failure_reason = (
            None
            if passed
            else "; ".join(f"{s.scorer_type}: {s.detail}" for s in scores if not s.passed)
        )

        # Surface the judge prompt/response (at most one judge per case) so the
        # store can write them to their dedicated columns in Phase 5.
        judge_prompt = judge_response = None
        for s in scores:
            if s.scorer_type == "judge" and "judge_prompt" in s.extra:
                judge_prompt = s.extra["judge_prompt"]
                judge_response = s.extra["judge_response"]

        return TrialResult(
            trial_idx=trial_idx,
            passed=passed,
            failure_reason=failure_reason,
            scores=[s.to_dict() for s in scores],
            judge_prompt=judge_prompt,
            judge_response=judge_response,
            final_text=result.final_text,
            tool_calls=[{"name": tc.name, "args": tc.args} for tc in result.tool_calls],
            raw_events=result.raw_events,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=cost,
        )


async def run_cases(
    cases: list[EvalCase],
    adapter: AgentAdapter,
    config: EvalGateConfig,
    trials_override: int | None = None,
) -> RunReport:
    """Run all eval cases and return a RunReport.

    Args:
        cases: Validated EvalCase objects from loader.load_cases().
        adapter: The agent wrapped in an AgentAdapter (e.g. ADKAdapter).
        config: Loaded EvalGateConfig (from evalgate.toml).
        trials_override: When set, overrides each case's trial count. Used by
                         `evalgate run --trials N`.

    Returns:
        RunReport containing CaseResult for each case.
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_trials)
    # A single-element list is the idiomatic way to share a mutable counter
    # across coroutines on the same event loop without threading primitives.
    llm_call_counter = [0]

    cost_per_input = config.costs.input_per_million / 1_000_000
    cost_per_output = config.costs.output_per_million / 1_000_000

    case_results: list[CaseResult] = []

    for case in cases:
        n_trials = trials_override if trials_override is not None else case.trials
        logger.info("Case %r: launching %d trial(s)", case.id, n_trials)

        # All trials for this case run concurrently, bounded by the semaphore.
        # asyncio.gather preserves order, so trial_results[i] = trial i.
        tasks = [
            _run_single_trial(
                case,
                i,
                adapter,
                config,
                semaphore,
                llm_call_counter,
                cost_per_input,
                cost_per_output,
            )
            for i in range(n_trials)
        ]
        trial_results: list[TrialResult] = list(await asyncio.gather(*tasks))

        passes = sum(1 for t in trial_results if t.passed)
        logger.info(
            "Case %r: %d/%d trials passed",
            case.id,
            passes,
            n_trials,
        )
        case_results.append(CaseResult(case_id=case.id, trials=trial_results))

    return RunReport(cases=case_results)
