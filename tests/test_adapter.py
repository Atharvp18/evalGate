"""Tests for the ADKAdapter and the safe calculator tool.

LLM calls are NOT made. The adapter is tested by injecting a stub Runner
that yields a scripted sequence of synthetic ADK Events.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from examples.sec_agent.agent import calculate
from google.genai import types as genai_types

from evalgate.adapter import AgentRunResult, ToolCall
from evalgate.adapters.adk import ADKAdapter

# ---------------------------------------------------------------------------
# Helpers to build stub ADK Events
# ---------------------------------------------------------------------------


def _make_function_call_event(name: str, args: dict) -> Any:
    """Return a MagicMock that behaves like an ADK Event carrying a function call."""
    fc = MagicMock()
    fc.name = name
    fc.args = args

    event = MagicMock()
    event.get_function_calls.return_value = [fc]
    event.get_function_responses.return_value = []
    event.is_final_response.return_value = False
    event.content = None
    event.usage_metadata = None
    event.author = "agent"
    return event


def _make_final_text_event(text: str, in_tokens: int = 0, out_tokens: int = 0) -> Any:
    event = MagicMock()
    event.get_function_calls.return_value = []
    event.get_function_responses.return_value = []
    event.is_final_response.return_value = True

    part = genai_types.Part(text=text)
    event.content = genai_types.Content(role="model", parts=[part])

    um = MagicMock()
    um.prompt_token_count = in_tokens
    um.candidates_token_count = out_tokens
    event.usage_metadata = um
    event.author = "agent"
    return event


def _async_iter(events: list[Any]):
    async def gen(**_kwargs):
        for e in events:
            yield e
    return gen


# ---------------------------------------------------------------------------
# ADKAdapter
# ---------------------------------------------------------------------------


class TestADKAdapter:
    def _make_adapter_with_events(self, events: list[Any]) -> ADKAdapter:
        agent = MagicMock()
        agent.name = "stub_agent"
        adapter = ADKAdapter.__new__(ADKAdapter)
        adapter._agent = agent
        adapter._app_name = "stub_agent"
        adapter._user_id = "evalgate"

        session_svc = MagicMock()

        async def create_session(**kwargs):
            return MagicMock(id=kwargs.get("session_id", "s1"))

        session_svc.create_session = create_session
        adapter._session_service = session_svc

        runner = MagicMock()
        runner.run_async = _async_iter(events)
        adapter._runner = runner
        return adapter

    def test_collects_tool_calls_in_order(self) -> None:
        events = [
            _make_function_call_event("lookup_cik", {"ticker": "NVDA"}),
            _make_function_call_event("get_company_facts", {"cik": "0001045810"}),
            _make_final_text_event("done"),
        ]
        adapter = self._make_adapter_with_events(events)

        result = asyncio.run(adapter.run("test query"))

        assert isinstance(result, AgentRunResult)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0] == ToolCall("lookup_cik", {"ticker": "NVDA"})
        assert result.tool_calls[1] == ToolCall("get_company_facts", {"cik": "0001045810"})

    def test_final_text_from_last_final_response_event(self) -> None:
        events = [
            _make_function_call_event("lookup_cik", {"ticker": "AAPL"}),
            _make_final_text_event("Apple's CIK is 0000320193."),
        ]
        adapter = self._make_adapter_with_events(events)
        result = asyncio.run(adapter.run("what is apple's cik?"))
        assert result.final_text == "Apple's CIK is 0000320193."

    def test_sums_token_counts_across_events(self) -> None:
        events = [
            _make_final_text_event("partial", in_tokens=10, out_tokens=5),
            _make_final_text_event("final", in_tokens=20, out_tokens=8),
        ]
        adapter = self._make_adapter_with_events(events)
        result = asyncio.run(adapter.run("q"))
        assert result.input_tokens == 30
        assert result.output_tokens == 13

    def test_records_latency(self) -> None:
        events = [_make_final_text_event("done")]
        adapter = self._make_adapter_with_events(events)
        result = asyncio.run(adapter.run("q"))
        assert result.latency_ms >= 0.0

    def test_raw_events_captured_per_event(self) -> None:
        events = [
            _make_function_call_event("lookup_cik", {"ticker": "MSFT"}),
            _make_final_text_event("ok"),
        ]
        adapter = self._make_adapter_with_events(events)
        result = asyncio.run(adapter.run("q"))
        assert len(result.raw_events) == 2
        assert result.raw_events[0]["function_calls"][0]["name"] == "lookup_cik"
        assert result.raw_events[1]["is_final"] is True
        assert result.raw_events[1]["text"] == "ok"

    def test_empty_final_text_when_no_final_event(self) -> None:
        events = [_make_function_call_event("x", {})]
        adapter = self._make_adapter_with_events(events)
        result = asyncio.run(adapter.run("q"))
        assert result.final_text == ""


# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------


class TestCalculate:
    def test_basic_arithmetic(self) -> None:
        r = calculate("3 + 4 * 2")
        assert r["result"] == 11.0
        assert r["error"] is None

    def test_growth_rate_expression(self) -> None:
        r = calculate("(81615 - 60922) / 60922 * 100")
        assert r["result"] == pytest.approx(33.96, abs=0.01)

    def test_negative_numbers(self) -> None:
        r = calculate("-5 + 3")
        assert r["result"] == -2.0

    def test_rejects_attribute_access(self) -> None:
        r = calculate("__import__('os').system('echo pwned')")
        assert r["result"] is None
        assert r["error"] is not None

    def test_rejects_name_lookup(self) -> None:
        r = calculate("x + 1")
        assert r["result"] is None
        assert r["error"] is not None

    def test_rejects_function_call(self) -> None:
        r = calculate("abs(-5)")
        assert r["result"] is None
        assert r["error"] is not None
