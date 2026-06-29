#!/usr/bin/env python3
"""One-shot script to record EDGAR fixtures for the 10 subject companies.

Run from the repo root:
    python examples/sec_agent/record_fixtures.py

Reads edgar.user_agent from evalgate.toml — update that first.
Fixtures are committed to git. Re-run only when EDGAR data drift is suspected
(the nightly smoke CI will flag that automatically).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script from the repo root without installing the package.
_repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

from examples.sec_agent.tools.edgar import EdgarClient, configure_client, lookup_cik

from evalgate.config import load_config

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "XOM"]


def main() -> None:
    cfg = load_config()
    cfg.validate()

    fixtures_dir = Path(cfg.edgar.fixtures_dir)
    client = EdgarClient(
        mode="live",
        record=True,
        fixtures_dir=fixtures_dir,
        user_agent=cfg.edgar.user_agent,
        requests_per_second=cfg.edgar.requests_per_second,
    )
    configure_client(client)

    print(f"Recording fixtures → {fixtures_dir.resolve()}")
    print(f"User-Agent : {cfg.edgar.user_agent}")
    print(f"Rate limit : {cfg.edgar.requests_per_second} req/s\n")

    # 1. Ticker map — one request covers all tickers.
    print("Fetching ticker map...")
    tickers_data = client.get_tickers()
    print(f"  {len(tickers_data):,} companies in SEC registry\n")

    # 2. For each company: companyfacts + submissions.
    for ticker in TICKERS:
        try:
            info = lookup_cik(ticker)
            cik = info["cik"]
            print(f"[{ticker}] CIK={cik}  {info['name']}")

            print("  → company facts ...")
            client.get_company_facts_raw(cik)

            print("  → submissions ...")
            client.get_submissions_raw(cik)

            print("  done\n")
        except Exception as exc:
            print(f"  ERROR: {exc}\n")

    print("All done. Review fixtures/ then commit.")
    client.close()


if __name__ == "__main__":
    main()
