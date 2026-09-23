"""Unit tests for runner.py's retry-on-transient-error logic. No real LLM
calls: a stub AgentAdapter raises the exceptions under test directly.
"""

from __future__ import annotations

import asyncio

import pytest

from evalgate.adapter import AgentRunResult
from evalgate.config import EvalGateConfig
from evalgate.runner import run_cases
from evalgate.schema import EvalCase


def _case(**overrides) -> EvalCase:
    base = {
        "id": "test_case",
        "description": "test",
        "input": "What is X?",
        "trials": 1,
        "expected": {"contains": ["ok"]},
        "scoring": [{"type": "contains"}],
    }
    base.update(overrides)
    return EvalCase(**base)


class _FlakyAdapter:
    """Raises `exc` on the first call, then succeeds — used to verify the
    runner's one-retry behavior for a specific exception message."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    async def run(self, query: str) -> AgentRunResult:
        self.calls += 1
        if self.calls == 1:
            raise self._exc
        return AgentRunResult(final_text="ok")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # The runner's retry path sleeps 60s between attempts — patch it out so
    # this test runs instantly instead of taking a real minute.
    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


def test_run_cases_retries_on_resource_exhausted():
    adapter = _FlakyAdapter(RuntimeError("429 RESOURCE_EXHAUSTED"))
    report = asyncio.run(run_cases([_case()], adapter, EvalGateConfig()))
    trial = report.cases[0].trials[0]
    assert trial.passed
    assert adapter.calls == 2


def test_run_cases_retries_on_transient_server_error_503():
    # Regression test: a Gemini 503 ("the service is currently unavailable")
    # used to fail the trial permanently on the first hit — no retry — even
    # though it's exactly the kind of transient error the retry path exists
    # for. See the RESOURCE_EXHAUSTED case above for the counterpart.
    adapter = _FlakyAdapter(RuntimeError("google.genai.errors.ServerError: 503 UNAVAILABLE"))
    report = asyncio.run(run_cases([_case()], adapter, EvalGateConfig()))
    trial = report.cases[0].trials[0]
    assert trial.passed
    assert adapter.calls == 2


def test_run_cases_does_not_retry_ordinary_errors():
    adapter = _FlakyAdapter(ValueError("some unrelated bug"))
    report = asyncio.run(run_cases([_case()], adapter, EvalGateConfig()))
    trial = report.cases[0].trials[0]
    assert not trial.passed
    assert adapter.calls == 1
