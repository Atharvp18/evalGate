"""SEC EDGAR HTTP client with live/replay/record modes, and ADK tool functions.

The client has two modes:
- live: real HTTP calls to EDGAR; optionally records responses as fixture files.
- replay: reads from fixture files only; never touches the network.

Eval runs always use replay so they are deterministic and don't hit rate limits.
The agent REPL and the nightly smoke test use live mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# XBRL us-gaap concepts kept during trimming.
_KEEP_CONCEPTS: frozenset[str] = frozenset(
    {
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "GrossProfit",
        "OperatingIncomeLoss",
        "Assets",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        "CommonStockSharesOutstanding",
    }
)

_MAX_ENTRIES_PER_CONCEPT = 12   # ~3 years of quarterly data for context trimming
_FIXTURE_SIZE_LIMIT_BYTES = 3 * 1024 * 1024  # 3 MB storage trim threshold


class FixtureMissError(Exception):
    """Raised in replay mode when no fixture file exists for a requested URL."""


class EdgarClient:
    """SEC EDGAR HTTP client.

    Args:
        mode: "live" makes real HTTP calls; "replay" reads from fixture files.
        fixtures_dir: Directory containing (or to receive) fixture JSON files.
        record: When True and mode="live", save every response as a fixture.
        user_agent: Required in live mode — SEC EDGAR policy mandates a contact address.
        requests_per_second: Live-mode politeness cap (EDGAR allows ~10; we default to 5).
    """

    _TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    _FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    _SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(
        self,
        mode: str = "replay",
        fixtures_dir: Path | str = "examples/sec_agent/fixtures",
        record: bool = False,
        user_agent: str = "",
        requests_per_second: int = 5,
    ) -> None:
        if mode not in ("live", "replay"):
            raise ValueError(f"mode must be 'live' or 'replay', got {mode!r}")
        self._mode = mode
        self._fixtures_dir = Path(fixtures_dir)
        self._record = record
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time: float = 0.0

        if mode == "live":
            if not user_agent:
                raise ValueError(
                    "user_agent is required in live mode. "
                    "SEC EDGAR policy requires a valid contact address in User-Agent."
                )
            # Only construct httpx.Client in live mode — replay mode must never
            # touch the network, so we enforce that by not creating the client at all.
            self._http = httpx.Client(
                headers={"User-Agent": user_agent},
                timeout=30.0,
                follow_redirects=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep if needed to stay under the requests-per-second limit."""
        elapsed = time.monotonic() - self._last_request_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def _fixture_path(self, url: str) -> Path:
        name = hashlib.sha256(url.encode()).hexdigest()[:16] + ".json"
        return self._fixtures_dir / name

    def _index_path(self) -> Path:
        return self._fixtures_dir / "index.json"

    def _load_index(self) -> dict[str, str]:
        p = self._index_path()
        if not p.exists():
            return {}
        with p.open() as f:
            return json.load(f)

    def _save_index(self, index: dict[str, str]) -> None:
        self._index_path().write_text(json.dumps(index, indent=2))

    def _save_fixture(self, url: str, data: dict) -> None:
        self._fixtures_dir.mkdir(parents=True, exist_ok=True)
        payload = data
        # Trim large company facts payloads before saving to stay under 3 MB.
        if len(json.dumps(data).encode()) > _FIXTURE_SIZE_LIMIT_BYTES and "facts" in data:
            payload = _trim_facts_for_storage(data)
            logger.info("Trimmed fixture for %s (raw response exceeded 3 MB)", url)
        path = self._fixture_path(url)
        path.write_text(json.dumps(payload, indent=2))
        index = self._load_index()
        index[url] = path.name
        self._save_index(index)

    def _get(self, url: str) -> dict:
        if self._mode == "replay":
            index = self._load_index()
            filename = index.get(url)
            if filename is None:
                raise FixtureMissError(
                    f"No fixture for URL: {url}\n"
                    f"Record it with: python examples/sec_agent/record_fixtures.py"
                )
            with (self._fixtures_dir / filename).open() as f:
                return json.load(f)

        # Live mode
        self._throttle()
        logger.debug("GET %s", url)
        resp = self._http.get(url)
        resp.raise_for_status()
        data: dict = resp.json()
        if self._record:
            self._save_fixture(url, data)
        return data

    # ------------------------------------------------------------------
    # Public methods (called by tool functions and record_fixtures.py)
    # ------------------------------------------------------------------

    def get_tickers(self) -> dict:
        """Fetch the full SEC ticker → CIK map."""
        return self._get(self._TICKERS_URL)

    def get_company_facts_raw(self, cik: str) -> dict:
        """Fetch raw XBRL company facts for a CIK (may be large)."""
        url = self._FACTS_URL.format(cik=_pad_cik(cik))
        return self._get(url)

    def get_submissions_raw(self, cik: str) -> dict:
        """Fetch raw submissions metadata for a CIK."""
        url = self._SUBMISSIONS_URL.format(cik=_pad_cik(cik))
        return self._get(url)

    def close(self) -> None:
        if self._mode == "live":
            self._http.close()


# ------------------------------------------------------------------
# Trimming helpers
# ------------------------------------------------------------------


def _pad_cik(cik: str | int) -> str:
    """Return a zero-padded 10-digit CIK string."""
    return f"{int(cik):010d}"


def _trim_facts_for_storage(data: dict) -> dict:
    """Trim a company facts payload to only the concepts EvalGate uses.

    Called when a raw fixture would exceed the 3 MB storage limit.
    Keeps the full historical series for retained concepts — only discards
    concepts we never read.
    """
    gaap = data.get("facts", {}).get("us-gaap", {})
    trimmed: dict[str, Any] = {c: gaap[c] for c in _KEEP_CONCEPTS if c in gaap}
    return {
        "cik": data.get("cik"),
        "entityName": data.get("entityName"),
        "facts": {"us-gaap": trimmed},
    }


def _trim_facts_for_context(data: dict) -> dict:
    """Trim a company facts payload for LLM context budget.

    Returns at most _MAX_ENTRIES_PER_CONCEPT recent 10-Q / 10-K entries per
    concept. Typical output is < 8 KB vs the raw 10 MB XBRL dump.

    The context-budget trim is separate from the storage trim: storage trim
    keeps all history for retained concepts; context trim then downsamples
    further so the LLM receives a small, parseable payload.
    """
    gaap = data.get("facts", {}).get("us-gaap", {})
    concepts: dict[str, list[dict]] = {}

    for concept in _KEEP_CONCEPTS:
        if concept not in gaap:
            continue
        units = gaap[concept].get("units", {})
        # Most financial facts are in USD; fall back to the first available unit.
        entries: list[dict] = units.get("USD") or next(iter(units.values()), [])
        # Keep only periodic filings (10-Q / 10-K), not point-in-time or amendments.
        entries = [e for e in entries if e.get("form") in ("10-Q", "10-K")]
        # Newest first, capped at _MAX_ENTRIES_PER_CONCEPT.
        entries = sorted(entries, key=lambda e: e.get("end", ""), reverse=True)
        entries = entries[:_MAX_ENTRIES_PER_CONCEPT]
        if entries:
            concepts[concept] = [
                {
                    "period_end": e.get("end"),
                    "value": e.get("val"),
                    "form": e.get("form"),
                    "filed": e.get("filed"),
                }
                for e in entries
            ]

    return {
        "entity": data.get("entityName"),
        "cik": _pad_cik(data.get("cik", 0)),
        "concepts": concepts,
    }


# ------------------------------------------------------------------
# Module-level client singleton — configured at agent startup.
# ------------------------------------------------------------------

_client: EdgarClient | None = None


def configure_client(client: EdgarClient) -> None:
    """Set the EdgarClient instance used by the tool functions below."""
    global _client
    _client = client


def _require_client() -> EdgarClient:
    if _client is None:
        raise RuntimeError(
            "EdgarClient not configured. Call configure_client() before using tools."
        )
    return _client


# ------------------------------------------------------------------
# ADK tool functions — docstrings are what the LLM agent reads.
# ------------------------------------------------------------------


def lookup_cik(ticker: str) -> dict:
    """Look up a company's SEC CIK number by its stock ticker symbol.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA" or "AAPL". Case-insensitive.

    Returns:
        dict with keys:
          - ticker (str): Normalised uppercase ticker.
          - cik (str): Zero-padded 10-digit CIK, e.g. "0001045810".
          - name (str): Official company name from the SEC registry.

    Raises:
        ValueError: If the ticker is not found in the SEC registry.
    """
    data = _require_client().get_tickers()
    ticker_upper = ticker.upper().strip()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return {
                "ticker": entry["ticker"],
                "cik": _pad_cik(entry["cik_str"]),
                "name": entry["title"],
            }
    raise ValueError(f"Ticker {ticker!r} not found in SEC EDGAR registry.")


def get_company_facts(cik: str) -> dict:
    """Retrieve key financial facts for a company from SEC EDGAR XBRL filings.

    Returns trimmed quarterly and annual values for revenue, net income,
    operating income, assets, EPS, and shares outstanding. At most 12 entries
    per concept (roughly 3 years of quarterly data) to fit in the model context.
    Always use lookup_cik() first to obtain the correct CIK.

    Args:
        cik: Zero-padded 10-digit CIK string, e.g. "0001045810" for Nvidia.

    Returns:
        dict with keys:
          - entity (str): Company name.
          - cik (str): Zero-padded CIK.
          - concepts (dict): Maps concept name → list of entries, each with
            period_end, value (in USD), form ("10-Q"/"10-K"), and filed date.
    """
    raw = _require_client().get_company_facts_raw(cik)
    return _trim_facts_for_context(raw)


def get_recent_filings(cik: str, form_type: str = "10-Q") -> dict:
    """Retrieve recent SEC filing metadata for a company.

    Use this to find which quarters have been filed and their accession numbers.
    For financial figures, prefer get_company_facts() which gives structured data.

    Args:
        cik: Zero-padded 10-digit CIK string.
        form_type: SEC form type to filter on. "10-Q" for quarterly (default),
                   "10-K" for annual reports.

    Returns:
        dict with keys:
          - entity (str): Company name.
          - cik (str): CIK as returned by EDGAR.
          - form_type (str): The form type filtered on.
          - filings (list): Up to 5 most recent filings, each with:
            accession, date, form, primary_doc.
    """
    raw = _require_client().get_submissions_raw(cik)
    recent = raw.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    filings = [
        {"accession": acc, "date": date, "form": form, "primary_doc": doc}
        for form, date, acc, doc in zip(forms, dates, accessions, docs, strict=False)
        if form == form_type
    ][:5]

    return {
        "entity": raw.get("name"),
        "cik": raw.get("cik"),
        "form_type": form_type,
        "filings": filings,
    }
