"""Interactive REPL for the SEC agent.

Run:
    python -m examples.sec_agent.chat

Uses LIVE EDGAR mode (real network) so you can ask about any ticker, not just
the 10 fixtured ones. Loads GOOGLE_API_KEY from .env.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running as `python -m examples.sec_agent.chat` from the repo root.
_repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv

load_dotenv(_repo_root / ".env", override=True)

from examples.sec_agent.agent import build_agent
from examples.sec_agent.tools.edgar import EdgarClient, configure_client

from evalgate.adapters.adk import ADKAdapter
from evalgate.config import load_config


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

    agent = build_agent()
    adapter = ADKAdapter(agent)

    print("SEC Agent REPL — type your question (Ctrl-D or 'quit' to exit)\n")
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            break

        try:
            result = await adapter.run(query)
        except Exception as exc:
            print(f"[error] {exc}\n")
            continue

        print(f"\nagent> {result.final_text}\n")
        if result.tool_calls:
            print(f"  ({len(result.tool_calls)} tool calls, "
                  f"{result.input_tokens}+{result.output_tokens} tokens, "
                  f"{result.latency_ms:.0f} ms)\n")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
