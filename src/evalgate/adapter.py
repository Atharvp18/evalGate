"""AgentAdapter Protocol and AgentRunResult — framework-agnostic seam for evalgate.

The core framework depends ONLY on this Protocol. Concrete adapters live in
`evalgate.adapters.<framework>` (ADK is the first one). To add support for a
new agent framework, implement `AgentAdapter` and return `AgentRunResult` —
no changes to runner.py, scorers, or stats are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation observed during an agent run.

    Attributes:
        name: The tool function name.
        args: The arguments passed to the tool. Adapters MUST stringify
              non-JSON-native values so trajectory scorers can do partial
              dict matching by value comparison.
    """

    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class AgentRunResult:
    """Everything one agent invocation produced — the input to all scorers.

    Attributes:
        final_text: The agent's final user-visible answer.
        tool_calls: Tool invocations in execution order.
        input_tokens: Prompt token count summed across all model calls.
        output_tokens: Completion token count summed across all model calls.
        latency_ms: Wall-clock time from adapter.run() start to return.
        raw_events: Adapter-specific event list, stored verbatim in the DB
                    so `mine-trace` can reconstruct the run later.
    """

    final_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw_events: list[dict] = field(default_factory=list)


class AgentAdapter(Protocol):
    """The seam between evalgate and any agent framework.

    Implementations MUST create a fresh, isolated session per call — no state
    must leak between trials. This is a correctness requirement: trials are
    intended to be independent samples; shared session memory would correlate
    them and invalidate the Wilson confidence interval.
    """

    async def run(self, query: str) -> AgentRunResult: ...
