"""The 10 regression injections used by the Phase 9 catch-rate study.

Each injection is applied as an in-process monkeypatch (module attribute
substitution) rather than an on-disk file patch. Both approaches produce the
same regressed agent behavior, but monkeypatching is strictly safer here:
`unittest.mock.patch` guarantees the original is restored even if a run
crashes mid-injection, so a killed study can never leave the real source tree
(or a committed fixture file) modified. On-disk patching would need its own
crash-safe revert logic to get the same guarantee.

Each `Injection` bundles:
  - `patch()`: an optional context manager for module-level state (prompts,
    trimming logic, the EDGAR client) that can't be expressed as a
    `build_agent()` argument.
  - `build_kwargs`: overrides passed straight into `build_agent()` (temperature,
    or a replacement tool list) for injections that ARE naturally expressible
    that way — no monkeypatching needed for those.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

from examples.sec_agent import agent as agent_module
from examples.sec_agent.tools import edgar as edgar_module

_NULL_CONTEXT = contextlib.nullcontext()


@dataclass
class Injection:
    number: int
    id: str
    description: str
    patch: Callable[[], contextlib.AbstractContextManager] | None = None
    build_kwargs: dict[str, Any] = field(default_factory=dict)

    def apply(self) -> contextlib.AbstractContextManager:
        return self.patch() if self.patch is not None else _NULL_CONTEXT


# ---------------------------------------------------------------------------
# #1 — report_agent stops being told to cite filings.
# ---------------------------------------------------------------------------

_REPORT_NO_CITATION = """\
You are the report agent. You produce the final user-facing answer.

Rules:
- NEVER invent numbers, dates, tickers, or CIKs. If a value was not retrieved,
  say so explicitly rather than approximating.
- Format large numbers with units ("$81.6 billion", not "81615000000").
- Keep the answer focused on what the user asked. No filler.
"""


@contextlib.contextmanager
def _no_citation_requirement() -> Iterator[None]:
    with mock.patch.object(agent_module, "REPORT_INSTRUCTION", _REPORT_NO_CITATION):
        yield


# ---------------------------------------------------------------------------
# #2 — retrieval_agent told to prefer 10-K (annual) totals when asked for a
# quarter.
# ---------------------------------------------------------------------------

_RETRIEVAL_PREFERS_ANNUAL = """\
You are the retrieval agent. Your job is to find SEC EDGAR data the team needs.

Workflow:
1. If the user mentions a company by ticker or name, call lookup_cik(ticker)
   to obtain the CIK number. Always do this first when you do not have a CIK.
2. Use get_company_facts(cik) to fetch trimmed quarterly/annual financials
   (revenue, net income, EPS, assets, shares). When the user asks about a
   quarter, PREFER the most recent annual (10-K) figures over quarterly
   (10-Q) figures — annual totals are more authoritative.
3. Use get_recent_filings(cik, form_type) only if the user asks about specific
   filings, accession numbers, or filing dates.

Rules:
- Fetch ONLY the companies and concepts the question requires. Do not pre-fetch
  comparison companies "in case" — that wastes context.
- Return the raw structured data you retrieved. Do NOT compute growth rates or
  comparisons yourself; that is the analysis agent's job.
- If a tool errors, report the error verbatim so the coordinator can decide
  whether to try a different ticker or stop.
"""


@contextlib.contextmanager
def _prefer_annual_over_quarterly() -> Iterator[None]:
    with mock.patch.object(agent_module, "RETRIEVAL_INSTRUCTION", _RETRIEVAL_PREFERS_ANNUAL):
        yield


# ---------------------------------------------------------------------------
# #3 — context trimming off-by-one: the second-most-recent quarter is
# returned as if it were the most recent.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _trim_off_by_one() -> Iterator[None]:
    original = edgar_module._trim_facts_for_context

    def _shifted(data: dict) -> dict:
        trimmed = original(data)
        for entries in trimmed.get("concepts", {}).values():
            if len(entries) > 1:
                entries.pop(0)  # drop the true most-recent entry
        return trimmed

    with mock.patch.object(edgar_module, "_trim_facts_for_context", _shifted):
        yield


# ---------------------------------------------------------------------------
# #4 — get_recent_filings dropped from the retrieval agent's tool list.
#
# Expressed as a build_agent() kwarg, not a patch: the tool list is a literal
# in build_agent()'s body, so there's no module-level list to monkeypatch.
# ---------------------------------------------------------------------------

_RETRIEVAL_TOOLS_NO_FILINGS = [agent_module.lookup_cik, agent_module.get_company_facts]


# ---------------------------------------------------------------------------
# #5 — agent temperature raised to 1.0 (a noise increase, not a logic bug —
# the spec expects this to be caught only by N-trial statistics, not any
# single trial).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# #6 — tool output truncated to 500 chars before returning to the model.
# `functools.wraps` keeps the name/docstring the LLM reads identical; only the
# runtime behavior changes, which is what makes this a realistic subtle bug
# rather than a tool the LLM would visibly avoid.
# ---------------------------------------------------------------------------


@functools.wraps(agent_module.get_company_facts)
def _truncated_get_company_facts(cik: str) -> dict:
    real = agent_module.get_company_facts(cik)
    serialized = json.dumps(real)
    if len(serialized) <= 500:
        return real
    return {"_truncated_output": serialized[:500]}


_RETRIEVAL_TOOLS_TRUNCATED = [
    agent_module.lookup_cik,
    _truncated_get_company_facts,
    agent_module.get_recent_filings,
]


# ---------------------------------------------------------------------------
# #7 — calculate() silently rounds to the nearest billion.
# ---------------------------------------------------------------------------


@functools.wraps(agent_module.calculate)
def _rounding_calculate(expression: str) -> dict:
    real = agent_module.calculate(expression)
    if real.get("result") is not None:
        real["result"] = round(real["result"] / 1_000_000_000) * 1_000_000_000
    return real


_ANALYSIS_TOOLS_ROUNDING = [_rounding_calculate]


# ---------------------------------------------------------------------------
# #8 — coordinator skips the analysis agent for comparison questions.
# ---------------------------------------------------------------------------

_COORDINATOR_SKIP_ANALYSIS = """\
You coordinate a team that answers questions about US public companies using
SEC EDGAR filings.

You have three sub-agents you can transfer to:
- retrieval_agent: fetches CIKs, company facts, and filings metadata.
- analysis_agent: does arithmetic on retrieved values.
- report_agent: writes the final cited answer.

Standard flow for a financial question:
1. Transfer to retrieval_agent to fetch the underlying data.
2. Always transfer to report_agent for the final answer, with the source
   data available in context. Skip analysis_agent even for comparison or
   growth-rate questions — report_agent can describe the numbers directly.
"""


@contextlib.contextmanager
def _skip_analysis_for_comparisons() -> Iterator[None]:
    with mock.patch.object(agent_module, "COORDINATOR_INSTRUCTION", _COORDINATOR_SKIP_ANALYSIS):
        yield


# ---------------------------------------------------------------------------
# #9 — one fixture's revenue value corrupted by +10% (a data regression).
#
# Patches EdgarClient.get_company_facts_raw at the class level so the
# corruption happens in memory, on the response the client returns — the
# committed fixture file on disk is never touched, so a crashed study run
# can never leave a corrupted fixture behind.
# ---------------------------------------------------------------------------

_CORRUPT_CIK = "0001045810"  # Nvidia
_REVENUE_CONCEPTS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")


@contextlib.contextmanager
def _corrupt_fixture_revenue() -> Iterator[None]:
    original = edgar_module.EdgarClient.get_company_facts_raw

    def _corrupted(self: edgar_module.EdgarClient, cik: str) -> dict:
        data = original(self, cik)
        if edgar_module._pad_cik(cik) != _CORRUPT_CIK:
            return data
        data = copy.deepcopy(data)
        gaap = data.get("facts", {}).get("us-gaap", {})
        for concept in _REVENUE_CONCEPTS:
            entries = gaap.get(concept, {}).get("units", {}).get("USD", [])
            if not entries:
                continue
            newest = max(entries, key=lambda e: e.get("end", ""))
            newest["val"] = round(newest["val"] * 1.1)
        return data

    with mock.patch.object(edgar_module.EdgarClient, "get_company_facts_raw", _corrupted):
        yield


# ---------------------------------------------------------------------------
# #10 — report_agent forced to answer in exactly one sentence
# (information-loss regression).
# ---------------------------------------------------------------------------

_REPORT_ONE_SENTENCE = """\
You are the report agent. You produce the final user-facing answer.

Rules:
- Answer in EXACTLY one sentence, no matter how many data points or filings
  are involved. Combine everything into that single sentence.
- Use the company name and the source filing for every figure when it fits.
- NEVER invent numbers, dates, tickers, or CIKs.
"""


@contextlib.contextmanager
def _one_sentence_report() -> Iterator[None]:
    with mock.patch.object(agent_module, "REPORT_INSTRUCTION", _REPORT_ONE_SENTENCE):
        yield


# ---------------------------------------------------------------------------
# The 10 injections, in spec order.
# ---------------------------------------------------------------------------

INJECTIONS: list[Injection] = [
    Injection(
        1, "no_citation_requirement", "Delete the citation requirement from report_agent",
        patch=_no_citation_requirement,
    ),
    Injection(
        2, "prefer_annual_over_quarterly",
        "retrieval_agent prefers 10-K annual totals when asked for quarterly data",
        patch=_prefer_annual_over_quarterly,
    ),
    Injection(
        3, "trim_off_by_one",
        "Context trimming returns the second-most-recent quarter as most recent",
        patch=_trim_off_by_one,
    ),
    Injection(
        4, "remove_recent_filings_tool", "get_recent_filings removed from retrieval_agent's tools",
        build_kwargs={"retrieval_tools": _RETRIEVAL_TOOLS_NO_FILINGS},
    ),
    Injection(
        5, "temperature_1_0", "Agent temperature raised to 1.0 (noise increase)",
        build_kwargs={"temperature": 1.0},
    ),
    Injection(
        6, "truncate_tool_output",
        "Tool output truncated to 500 chars before returning to the model",
        build_kwargs={"retrieval_tools": _RETRIEVAL_TOOLS_TRUNCATED},
    ),
    Injection(
        7, "calculate_rounds_to_billion", "calculate() silently rounds to the nearest billion",
        build_kwargs={"analysis_tools": _ANALYSIS_TOOLS_ROUNDING},
    ),
    Injection(
        8, "skip_analysis_for_comparisons",
        "Coordinator skips the analysis agent for comparison questions",
        patch=_skip_analysis_for_comparisons,
    ),
    Injection(
        9, "corrupt_fixture_revenue", "One fixture's revenue value corrupted by +10%",
        patch=_corrupt_fixture_revenue,
    ),
    Injection(
        10, "one_sentence_report", "report_agent forced to answer in exactly one sentence",
        patch=_one_sentence_report,
    ),
]
