"""ADKAdapter — wraps google.adk Runner, normalises events into AgentRunResult.

Every call to `run()` creates a fresh InMemorySession so trials are independent.
Function-call events are normalised to `ToolCall`; the final text answer is
assembled from the last `is_final_response()` event's content parts; token
counts are summed across every event that carries `usage_metadata`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from evalgate.adapter import AgentRunResult, ToolCall


class ADKAdapter:
    """Adapts a google.adk agent (or coordinator with sub-agents) to AgentAdapter.

    Args:
        agent: An ADK agent (typically the root coordinator).
        app_name: ADK requires an app name for session keying. Defaults to the agent name.
        user_id: Stable user id for the session service. Sessions are per-call,
                 so the user id only needs to be valid, not unique per trial.
    """

    def __init__(
        self,
        agent: BaseAgent,
        app_name: str | None = None,
        user_id: str = "evalgate",
    ) -> None:
        self._agent = agent
        self._app_name = app_name or agent.name
        self._user_id = user_id
        # Session service is shared across trials, but each trial creates a
        # NEW session — see _new_session() in run().
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=agent,
            session_service=self._session_service,
            app_name=self._app_name,
        )

    async def run(self, query: str) -> AgentRunResult:
        # Fresh session per call — correctness requirement (see adapter.py).
        # A random session_id guarantees no collision with prior trials in the
        # same InMemorySessionService.
        session_id = f"trial-{uuid.uuid4().hex}"
        await self._session_service.create_session(
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=session_id,
        )

        tool_calls: list[ToolCall] = []
        final_text = ""
        input_tokens = 0
        output_tokens = 0
        raw_events: list[dict] = []

        message = genai_types.Content(
            role="user", parts=[genai_types.Part(text=query)]
        )

        start = time.perf_counter()

        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=session_id,
            new_message=message,
        ):
            # 1) Tool calls — every function call in the event.
            for fc in event.get_function_calls():
                tool_calls.append(
                    ToolCall(name=fc.name or "", args=dict(fc.args or {}))
                )

            # 2) Final answer — the last is_final_response event's text.
            if event.is_final_response() and event.content:
                parts = event.content.parts or []
                final_text = "".join(p.text or "" for p in parts if p.text)

            # 3) Token usage — sum across every event that reports it.
            um = event.usage_metadata
            if um is not None:
                input_tokens += um.prompt_token_count or 0
                output_tokens += um.candidates_token_count or 0

            # 4) Raw events — store a lightweight dict snapshot for mine-trace.
            raw_events.append(_event_to_dict(event))

        latency_ms = (time.perf_counter() - start) * 1000.0

        return AgentRunResult(
            final_text=final_text.strip(),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            raw_events=raw_events,
        )


def _event_to_dict(event: Any) -> dict:
    """Convert an ADK Event to a JSON-serialisable dict for storage.

    We deliberately keep only the fields we may need later for trace mining;
    the full Event object contains many internal references that don't
    round-trip cleanly to JSON.
    """
    function_calls = [
        {"name": fc.name, "args": dict(fc.args or {})}
        for fc in event.get_function_calls()
    ]
    function_responses = [
        {"name": fr.name, "response": _safe_jsonable(fr.response)}
        for fr in event.get_function_responses()
    ]
    text = ""
    if event.content and event.content.parts:
        text = "".join(p.text or "" for p in event.content.parts if p.text)
    return {
        "author": getattr(event, "author", None),
        "is_final": event.is_final_response(),
        "text": text,
        "function_calls": function_calls,
        "function_responses": function_responses,
    }


def _safe_jsonable(value: Any) -> Any:
    """Best-effort JSON-friendly coercion for tool response payloads."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_safe_jsonable(v) for v in value]
    return repr(value)
