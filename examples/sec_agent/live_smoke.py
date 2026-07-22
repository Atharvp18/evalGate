"""One-shot live-mode sanity check for the nightly-smoke CI workflow.

Asks the agent a single real question against the live SEC EDGAR API (not
replay/fixtures) and asserts a non-empty answer comes back. This is the only
thing that can catch EDGAR API drift — the PR gate runs entirely in replay
mode against frozen fixtures and would never notice a live-endpoint change.

Run:
    python examples/sec_agent/live_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv

load_dotenv(_repo_root / ".env", override=True)

from evalgate.adapters.adk import ADKAdapter
from evalgate.config import load_config
from examples.sec_agent.agent import build_agent
from examples.sec_agent.tools.edgar import EdgarClient, configure_client

QUESTION = "What was Nvidia's revenue in its most recent quarter?"


async def main() -> None:
    cfg = load_config()
    cfg.validate()

    client = EdgarClient(
        mode="live",
        record=False,
        fixtures_dir=cfg.edgar.fixtures_dir,
        user_agent=cfg.edgar.user_agent,
        requests_per_second=cfg.edgar.requests_per_second,
    )
    configure_client(client)

    adapter = ADKAdapter(build_agent())
    result = await adapter.run(QUESTION)
    client.close()

    print(f"Q: {QUESTION}")
    print(f"A: {result.final_text}")
    print(f"({len(result.tool_calls)} tool calls, {result.latency_ms:.0f} ms)")

    assert result.final_text.strip(), "agent returned an empty answer — EDGAR API may have drifted"


if __name__ == "__main__":
    asyncio.run(main())
