"""Unit tests for loader.py — YAML discovery, validation, and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalgate.loader import load_cases
from evalgate.schema import EvalCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_case(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


VALID_JUDGE_YAML = """\
id: my_case
description: A test case
input: What is X?
scoring:
  - type: judge
    rubric: The answer is correct.
"""

VALID_NUMERIC_YAML = """\
id: numeric_case
description: A numeric case
input: What is the revenue?
expected:
  numeric:
    value: 1000000.0
    tolerance_pct: 2.0
scoring:
  - type: numeric
"""

VALID_CONTAINS_YAML = """\
id: contains_case
description: A contains case
input: Name a company.
expected:
  contains: [Apple]
scoring:
  - type: contains
"""


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoadCasesHappyPath:
    def test_loads_single_yaml_file(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "my_case.yaml", VALID_JUDGE_YAML)
        cases = load_cases(tmp_path)
        assert len(cases) == 1
        assert cases[0].id == "my_case"
        assert isinstance(cases[0], EvalCase)

    def test_loads_multiple_yaml_files(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "a.yaml", VALID_JUDGE_YAML)
        _write_case(tmp_path / "b.yaml", VALID_NUMERIC_YAML)
        cases = load_cases(tmp_path)
        assert len(cases) == 2
        # Loader sorts by id.
        assert cases[0].id == "my_case"
        assert cases[1].id == "numeric_case"

    def test_also_loads_yml_extension(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "my_case.yml", VALID_JUDGE_YAML)
        cases = load_cases(tmp_path)
        assert len(cases) == 1

    def test_sorted_by_id(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "z.yaml", VALID_NUMERIC_YAML)
        _write_case(tmp_path / "a.yaml", VALID_CONTAINS_YAML)
        _write_case(tmp_path / "m.yaml", VALID_JUDGE_YAML)
        cases = load_cases(tmp_path)
        ids = [c.id for c in cases]
        assert ids == sorted(ids)

    def test_accepts_path_str(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "my_case.yaml", VALID_JUDGE_YAML)
        cases = load_cases(str(tmp_path))
        assert len(cases) == 1

    def test_ignores_non_yaml_files(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "my_case.yaml", VALID_JUDGE_YAML)
        (tmp_path / "notes.txt").write_text("ignore me")
        (tmp_path / "README.md").write_text("# ignore")
        cases = load_cases(tmp_path)
        assert len(cases) == 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestLoadCasesErrors:
    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_cases(tmp_path / "does_not_exist")

    def test_empty_directory_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No YAML"):
            load_cases(tmp_path)

    def test_invalid_yaml_syntax_raises_value_error(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "bad.yaml", "id: foo\n  bad_indent: [unclosed")
        with pytest.raises(ValueError, match="YAML parse error"):
            load_cases(tmp_path)

    def test_non_mapping_yaml_raises_value_error(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "bad.yaml", "- item1\n- item2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_cases(tmp_path)

    def test_validation_error_includes_filename(self, tmp_path: Path) -> None:
        # Missing 'input' field — should fail validation.
        _write_case(
            tmp_path / "broken.yaml",
            "id: broken_case\ndescription: d\nscoring:\n  - type: judge\n    rubric: ok\n",
        )
        with pytest.raises(ValueError, match="broken.yaml"):
            load_cases(tmp_path)

    def test_duplicate_ids_raises_value_error(self, tmp_path: Path) -> None:
        _write_case(tmp_path / "a.yaml", VALID_JUDGE_YAML)
        # Same id "my_case" in a different file.
        _write_case(tmp_path / "b.yaml", VALID_JUDGE_YAML)
        with pytest.raises(ValueError, match="Duplicate case id"):
            load_cases(tmp_path)

    def test_extra_field_in_yaml_raises_value_error(self, tmp_path: Path) -> None:
        yaml_with_extra = VALID_JUDGE_YAML + "unknown_field: oops\n"
        _write_case(tmp_path / "bad.yaml", yaml_with_extra)
        with pytest.raises(ValueError, match="bad.yaml"):
            load_cases(tmp_path)

    def test_judge_without_rubric_raises_value_error(self, tmp_path: Path) -> None:
        _write_case(
            tmp_path / "bad.yaml",
            "id: bad\ndescription: d\ninput: q\nscoring:\n  - type: judge\n",
        )
        with pytest.raises(ValueError, match="bad.yaml"):
            load_cases(tmp_path)


# ---------------------------------------------------------------------------
# Full case round-trip (loader → EvalCase fields intact)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_all_fields_preserved(self, tmp_path: Path) -> None:
        yaml_content = """\
id: full_case
description: Full round-trip test
input: What was Nvidia revenue?
tags: [retrieval, numeric]
trials: 8
expected:
  numeric:
    value: 81615000000
    tolerance_pct: 2.0
  contains: [Nvidia]
  trajectory:
    - tool: lookup_cik
    - tool: get_company_facts
      args_contain: {cik: "0001045810"}
scoring:
  - type: numeric
  - type: trajectory
  - type: judge
    rubric: States revenue correctly.
pass_policy: all
thresholds:
  pass_rate_min: 0.8
"""
        _write_case(tmp_path / "full.yaml", yaml_content)
        cases = load_cases(tmp_path)
        c = cases[0]

        assert c.id == "full_case"
        assert c.trials == 8
        assert c.tags == ["retrieval", "numeric"]
        assert c.expected.numeric is not None
        assert c.expected.numeric.value == pytest.approx(81615000000)
        assert c.expected.contains == ["Nvidia"]
        assert len(c.expected.trajectory) == 2
        assert c.expected.trajectory[1].args_contain == {"cik": "0001045810"}
        assert len(c.scoring) == 3
        assert c.scoring[2].rubric == "States revenue correctly."
        assert c.thresholds.pass_rate_min == pytest.approx(0.8)
