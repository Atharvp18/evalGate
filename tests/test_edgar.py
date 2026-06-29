"""Unit tests for EdgarClient — replay mode, fixture recording, and facts trimming."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from examples.sec_agent.tools.edgar import (
    EdgarClient,
    FixtureMissError,
    _trim_facts_for_context,
    _trim_facts_for_storage,
    configure_client,
    lookup_cik,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fixture(fixtures_dir: Path, url: str, data: dict) -> None:
    """Write a fixture file and update index.json."""
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(url.encode()).hexdigest()[:16] + ".json"
    (fixtures_dir / name).write_text(json.dumps(data))
    index_path = fixtures_dir / "index.json"
    index: dict[str, str] = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[url] = name
    index_path.write_text(json.dumps(index))


def _make_facts_payload(concepts: list[str], n_entries: int = 20) -> dict:
    """Build a synthetic company facts JSON for testing."""
    entries: list[dict[str, Any]] = [
        {
            "end": f"202{i // 4}-{(i % 4) * 3 + 1:02d}-01",
            "val": (i + 1) * 1_000_000,
            "form": "10-Q",
            "filed": f"202{i // 4}-{(i % 4) * 3 + 2:02d}-01",
        }
        for i in range(n_entries)
    ]
    gaap = {c: {"units": {"USD": entries}} for c in concepts}
    return {"cik": 12345, "entityName": "Fake Corp", "facts": {"us-gaap": gaap}}


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------


class TestReplayMode:
    def test_hit_returns_stored_data(self, tmp_path: Path) -> None:
        url = "https://example.com/data.json"
        expected = {"entity": "Acme", "value": 42}
        _write_fixture(tmp_path, url, expected)

        client = EdgarClient(mode="replay", fixtures_dir=tmp_path)
        assert client._get(url) == expected

    def test_miss_raises_fixture_miss_error(self, tmp_path: Path) -> None:
        client = EdgarClient(mode="replay", fixtures_dir=tmp_path)
        with pytest.raises(FixtureMissError, match="record_fixtures"):
            client._get("https://example.com/missing.json")

    def test_replay_has_no_http_client_attribute(self, tmp_path: Path) -> None:
        """Replay mode must not construct an httpx client — no network access allowed."""
        client = EdgarClient(mode="replay", fixtures_dir=tmp_path)
        assert not hasattr(client, "_http")

    def test_empty_fixtures_dir_raises_fixture_miss_error(self, tmp_path: Path) -> None:
        client = EdgarClient(mode="replay", fixtures_dir=tmp_path / "empty")
        with pytest.raises(FixtureMissError):
            client._get("https://example.com/anything.json")


# ---------------------------------------------------------------------------
# Live + record mode
# ---------------------------------------------------------------------------


class TestLiveRecordMode:
    def test_record_saves_fixture_and_updates_index(self, tmp_path: Path) -> None:
        fake_data = {"entity": "Test Corp", "revenue": 999}

        mock_response = MagicMock()
        mock_response.json.return_value = fake_data
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client.get", return_value=mock_response):
            client = EdgarClient(
                mode="live",
                record=True,
                fixtures_dir=tmp_path,
                user_agent="EvalGate test <test@example.com>",
            )
            result = client._get("https://example.com/test.json")
            client.close()

        assert result == fake_data
        index = json.loads((tmp_path / "index.json").read_text())
        assert "https://example.com/test.json" in index

    def test_live_mode_requires_user_agent(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="user_agent"):
            EdgarClient(mode="live", fixtures_dir=tmp_path, user_agent="")

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mode"):
            EdgarClient(mode="streaming", fixtures_dir=tmp_path)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Facts trimming — storage trim
# ---------------------------------------------------------------------------


class TestTrimForStorage:
    def test_keeps_known_concepts(self) -> None:
        data = _make_facts_payload(["Revenues", "SomeObscureConcept"])
        result = _trim_facts_for_storage(data)
        gaap = result["facts"]["us-gaap"]
        assert "Revenues" in gaap
        assert "SomeObscureConcept" not in gaap

    def test_preserves_entity_and_cik(self) -> None:
        data = _make_facts_payload(["Revenues"])
        result = _trim_facts_for_storage(data)
        assert result["entityName"] == "Fake Corp"
        assert result["cik"] == 12345

    def test_keeps_all_history_for_retained_concepts(self) -> None:
        """Storage trim keeps full history; only context trim downsamples."""
        data = _make_facts_payload(["Revenues"], n_entries=50)
        result = _trim_facts_for_storage(data)
        entries = result["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
        assert len(entries) == 50


# ---------------------------------------------------------------------------
# Facts trimming — context trim
# ---------------------------------------------------------------------------


class TestTrimForContext:
    def test_keeps_known_concepts_drops_unknown(self) -> None:
        data = _make_facts_payload(["Revenues", "WeirdCustomConcept"])
        result = _trim_facts_for_context(data)
        assert "Revenues" in result["concepts"]
        assert "WeirdCustomConcept" not in result["concepts"]

    def test_caps_entries_at_max(self) -> None:
        data = _make_facts_payload(["Revenues"], n_entries=30)
        result = _trim_facts_for_context(data)
        assert len(result["concepts"]["Revenues"]) <= 12

    def test_entries_sorted_newest_first(self) -> None:
        data = _make_facts_payload(["NetIncomeLoss"], n_entries=10)
        result = _trim_facts_for_context(data)
        dates = [e["period_end"] for e in result["concepts"]["NetIncomeLoss"]]
        assert dates == sorted(dates, reverse=True)

    def test_output_has_expected_top_level_keys(self) -> None:
        data = _make_facts_payload(["Revenues"])
        result = _trim_facts_for_context(data)
        assert set(result.keys()) == {"entity", "cik", "concepts"}

    def test_each_entry_has_required_fields(self) -> None:
        data = _make_facts_payload(["Revenues"])
        result = _trim_facts_for_context(data)
        for entry in result["concepts"]["Revenues"]:
            assert set(entry.keys()) == {"period_end", "value", "form", "filed"}

    def test_filters_out_non_periodic_forms(self) -> None:
        """Only 10-Q and 10-K entries should survive the context trim."""
        entries = [
            {"end": "2024-01-01", "val": 100, "form": "8-K", "filed": "2024-01-05"},
            {"end": "2024-04-01", "val": 200, "form": "10-Q", "filed": "2024-04-05"},
        ]
        data = {
            "cik": 1,
            "entityName": "Corp",
            "facts": {"us-gaap": {"Revenues": {"units": {"USD": entries}}}},
        }
        result = _trim_facts_for_context(data)
        assert len(result["concepts"]["Revenues"]) == 1
        assert result["concepts"]["Revenues"][0]["form"] == "10-Q"

    def test_no_known_concepts_returns_empty_concepts(self) -> None:
        data = _make_facts_payload(["CompletelyUnknown"])
        result = _trim_facts_for_context(data)
        assert result["concepts"] == {}


# ---------------------------------------------------------------------------
# lookup_cik tool function
# ---------------------------------------------------------------------------


class TestLookupCik:
    def _make_ticker_client(self, tmp_path: Path, ticker_data: dict) -> EdgarClient:
        url = "https://www.sec.gov/files/company_tickers.json"
        _write_fixture(tmp_path, url, ticker_data)
        return EdgarClient(mode="replay", fixtures_dir=tmp_path)

    def test_finds_ticker_case_insensitive(self, tmp_path: Path) -> None:
        tickers = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
        client = self._make_ticker_client(tmp_path, tickers)
        configure_client(client)

        result = lookup_cik("nvda")
        assert result["ticker"] == "NVDA"
        assert result["cik"] == "0001045810"
        assert result["name"] == "NVIDIA CORP"

    def test_unknown_ticker_raises_value_error(self, tmp_path: Path) -> None:
        tickers = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
        client = self._make_ticker_client(tmp_path, tickers)
        configure_client(client)

        with pytest.raises(ValueError, match="ZZZZ"):
            lookup_cik("ZZZZ")

    def test_cik_is_zero_padded_to_ten_digits(self, tmp_path: Path) -> None:
        tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        client = self._make_ticker_client(tmp_path, tickers)
        configure_client(client)

        result = lookup_cik("AAPL")
        assert result["cik"] == "0000320193"
        assert len(result["cik"]) == 10
