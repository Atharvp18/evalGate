"""Unit tests for study/injections.py — verifies each injection actually changes
the intended behavior while active, and reverts cleanly (even on exception)
once the context manager exits. No LLM calls; these test the monkeypatch
mechanics and the wrapped tool/trim functions directly.
"""

from __future__ import annotations

import pytest
from examples.sec_agent import agent as agent_module
from examples.sec_agent.tools import edgar as edgar_module
from study.injections import INJECTIONS, Injection


def _get(injection_id: str) -> Injection:
    return next(i for i in INJECTIONS if i.id == injection_id)


# ---------------------------------------------------------------------------
# Every injection is well-formed and uniquely numbered/named.
# ---------------------------------------------------------------------------


def test_ten_injections_uniquely_numbered_and_named():
    assert len(INJECTIONS) == 10
    assert sorted(i.number for i in INJECTIONS) == list(range(1, 11))
    assert len({i.id for i in INJECTIONS}) == 10


def test_every_injection_has_a_patch_or_build_kwargs():
    for injection in INJECTIONS:
        assert injection.patch is not None or injection.build_kwargs


# ---------------------------------------------------------------------------
# Instruction-patching injections revert cleanly, including on exception.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection_id,module,attr",
    [
        ("no_citation_requirement", agent_module, "REPORT_INSTRUCTION"),
        ("prefer_annual_over_quarterly", agent_module, "RETRIEVAL_INSTRUCTION"),
        ("skip_analysis_for_comparisons", agent_module, "COORDINATOR_INSTRUCTION"),
        ("one_sentence_report", agent_module, "REPORT_INSTRUCTION"),
    ],
)
def test_instruction_patch_changes_and_reverts(injection_id, module, attr):
    original = getattr(module, attr)
    injection = _get(injection_id)

    with injection.apply():
        assert getattr(module, attr) != original

    assert getattr(module, attr) == original


def test_instruction_patch_reverts_even_on_exception():
    original = agent_module.REPORT_INSTRUCTION
    injection = _get("no_citation_requirement")

    with pytest.raises(RuntimeError), injection.apply():
        assert original != agent_module.REPORT_INSTRUCTION
        raise RuntimeError("boom")

    assert original == agent_module.REPORT_INSTRUCTION


def test_no_citation_requirement_drops_citation_text():
    injection = _get("no_citation_requirement")
    with injection.apply():
        assert "cite" not in agent_module.REPORT_INSTRUCTION.lower()


def test_prefer_annual_tells_agent_to_prefer_10k():
    injection = _get("prefer_annual_over_quarterly")
    with injection.apply():
        assert "10-K" in agent_module.RETRIEVAL_INSTRUCTION
        assert "PREFER" in agent_module.RETRIEVAL_INSTRUCTION


def test_skip_analysis_removes_step_2_transfer():
    injection = _get("skip_analysis_for_comparisons")
    with injection.apply():
        assert "Skip analysis_agent" in agent_module.COORDINATOR_INSTRUCTION


def test_one_sentence_report_demands_single_sentence():
    injection = _get("one_sentence_report")
    with injection.apply():
        assert "EXACTLY one sentence" in agent_module.REPORT_INSTRUCTION


# ---------------------------------------------------------------------------
# Trim off-by-one (#3): drops the newest entry per concept.
# ---------------------------------------------------------------------------


def test_trim_off_by_one_drops_newest_entry():
    raw_facts = {
        "cik": 1045810,
        "entityName": "NVIDIA CORP",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"end": "2026-01-31", "val": 100, "form": "10-Q", "filed": "x"},
                            {"end": "2026-04-26", "val": 200, "form": "10-Q", "filed": "x"},
                        ]
                    }
                }
            }
        },
    }
    normal = edgar_module._trim_facts_for_context(raw_facts)
    assert normal["concepts"]["Revenues"][0]["value"] == 200  # newest first, unpatched

    injection = _get("trim_off_by_one")
    with injection.apply():
        shifted = edgar_module._trim_facts_for_context(raw_facts)
    assert shifted["concepts"]["Revenues"][0]["value"] == 100  # newest dropped

    # Reverted afterward.
    normal_again = edgar_module._trim_facts_for_context(raw_facts)
    assert normal_again["concepts"]["Revenues"][0]["value"] == 200


def test_trim_off_by_one_leaves_single_entry_concepts_alone():
    raw_facts = {
        "cik": 1,
        "entityName": "X",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": [{"end": "2026-01-31", "val": 1, "form": "10-Q"}]}
                }
            }
        },
    }
    injection = _get("trim_off_by_one")
    with injection.apply():
        trimmed = edgar_module._trim_facts_for_context(raw_facts)
    # Only one entry existed — nothing to shift, so it must survive, not vanish.
    assert trimmed["concepts"]["Revenues"][0]["value"] == 1


# ---------------------------------------------------------------------------
# Tool-list-override injections (#4, #6, #7): expressed as build_kwargs.
# ---------------------------------------------------------------------------


def test_remove_recent_filings_drops_only_that_tool():
    injection = _get("remove_recent_filings_tool")
    tools = injection.build_kwargs["retrieval_tools"]
    names = {t.__name__ for t in tools}
    assert names == {"lookup_cik", "get_company_facts"}


def test_temperature_injection_overrides_to_one():
    injection = _get("temperature_1_0")
    assert injection.build_kwargs["temperature"] == 1.0


def test_truncated_tool_keeps_original_name_and_docstring():
    injection = _get("truncate_tool_output")
    tools = {t.__name__: t for t in injection.build_kwargs["retrieval_tools"]}
    truncated = tools["get_company_facts"]
    assert truncated.__doc__ == agent_module.get_company_facts.__doc__


def test_truncated_tool_truncates_large_output(monkeypatch):
    huge_result = {"entity": "X", "cik": "0000000001", "concepts": {"Revenues": ["x"] * 500}}
    monkeypatch.setattr(agent_module, "get_company_facts", lambda cik: huge_result)

    from study.injections import _truncated_get_company_facts

    result = _truncated_get_company_facts("0000000001")
    assert "_truncated_output" in result
    assert len(result["_truncated_output"]) == 500


def test_truncated_tool_passes_through_small_output(monkeypatch):
    small_result = {"entity": "X", "cik": "0000000001", "concepts": {}}
    monkeypatch.setattr(agent_module, "get_company_facts", lambda cik: small_result)

    from study.injections import _truncated_get_company_facts

    assert _truncated_get_company_facts("0000000001") == small_result


def test_rounding_calculate_rounds_to_nearest_billion():
    injection = _get("calculate_rounds_to_billion")
    rounding_calc = injection.build_kwargs["analysis_tools"][0]
    result = rounding_calc("1_600_000_000 + 200_000_000")
    assert result["result"] == 2_000_000_000


def test_rounding_calculate_preserves_errors():
    injection = _get("calculate_rounds_to_billion")
    rounding_calc = injection.build_kwargs["analysis_tools"][0]
    result = rounding_calc("not an expression")
    assert result["result"] is None
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# Fixture corruption (#9): only the targeted CIK's revenue is inflated.
# ---------------------------------------------------------------------------


def test_corrupt_fixture_only_affects_target_cik(monkeypatch):
    def fake_raw(self, cik):
        return {
            "cik": int(cik),
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [{"end": "2026-04-26", "val": 1000, "form": "10-Q"}]
                        }
                    }
                }
            },
        }

    monkeypatch.setattr(edgar_module.EdgarClient, "get_company_facts_raw", fake_raw)

    injection = _get("corrupt_fixture_revenue")
    client = edgar_module.EdgarClient.__new__(edgar_module.EdgarClient)  # no __init__ needed

    with injection.apply():
        corrupted = edgar_module.EdgarClient.get_company_facts_raw(client, "0001045810")
        untouched = edgar_module.EdgarClient.get_company_facts_raw(client, "0000320193")

    corrupted_val = corrupted["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["val"]
    untouched_val = untouched["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["val"]
    assert corrupted_val == 1100  # +10%
    assert untouched_val == 1000  # unaffected — different CIK


def test_corrupt_fixture_reverts_after_context(monkeypatch):
    def fake_raw(self, cik):
        return {"cik": int(cik), "facts": {}}

    monkeypatch.setattr(edgar_module.EdgarClient, "get_company_facts_raw", fake_raw)
    original = edgar_module.EdgarClient.get_company_facts_raw

    injection = _get("corrupt_fixture_revenue")
    with injection.apply():
        pass

    assert edgar_module.EdgarClient.get_company_facts_raw is original
