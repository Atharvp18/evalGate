"""Discovers, parses, and validates YAML eval case files from a directory."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from evalgate.schema import EvalCase

logger = logging.getLogger(__name__)


def load_case_file(path: Path) -> EvalCase:
    """Parse and validate a single YAML case file.

    Raises:
        ValueError: If the YAML is malformed or fails schema validation.
    """
    try:
        with path.open() as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error in {path.name}: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: expected a YAML mapping at the top level")

    try:
        return EvalCase(**raw)
    except ValidationError as e:
        # Re-raise with the filename so the user knows which file to fix.
        raise ValueError(f"Validation error in {path.name}:\n{e}") from e


def load_cases(cases_dir: Path | str) -> list[EvalCase]:
    """Load and validate all YAML eval case files from a directory.

    Args:
        cases_dir: Directory containing `.yaml` or `.yml` case files.

    Returns:
        List of validated EvalCase objects, sorted by id.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no YAML files are found, a file is invalid, or
                    duplicate case ids are detected.
    """
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")
    if not cases_dir.is_dir():
        raise ValueError(f"cases_dir must be a directory, got: {cases_dir}")

    yaml_files = sorted(cases_dir.glob("*.yaml")) + sorted(cases_dir.glob("*.yml"))
    # Deduplicate in case a file appears under both extensions (unlikely but safe).
    seen_paths: set[Path] = set()
    unique_files = [p for p in yaml_files if not (p in seen_paths or seen_paths.add(p))]  # type: ignore[func-returns-value]

    if not unique_files:
        raise ValueError(f"No YAML case files found in {cases_dir}")

    cases: list[EvalCase] = []
    seen_ids: dict[str, Path] = {}

    for path in unique_files:
        logger.debug("Loading case file: %s", path.name)
        case = load_case_file(path)

        if case.id in seen_ids:
            raise ValueError(
                f"Duplicate case id {case.id!r} — found in both "
                f"{seen_ids[case.id].name} and {path.name}"
            )
        seen_ids[case.id] = path
        cases.append(case)
        logger.debug("Loaded case %r from %s", case.id, path.name)

    cases.sort(key=lambda c: c.id)
    logger.info("Loaded %d eval case(s) from %s", len(cases), cases_dir)
    return cases
