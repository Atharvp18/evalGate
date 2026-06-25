"""Loads evalgate.toml from the working directory and exposes a typed config object."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CostsConfig:
    input_per_million: float = 0.075
    output_per_million: float = 0.30


@dataclass
class EdgarConfig:
    user_agent: str = "EvalGate research project <your-email@example.com>"
    mode: str = "replay"
    fixtures_dir: str = "examples/sec_agent/fixtures"
    requests_per_second: int = 5


@dataclass
class EvalGateConfig:
    trials: int = 6
    max_concurrent_trials: int = 4
    trial_timeout_s: int = 120
    judge_model: str = "gemini/gemini-2.0-flash"
    max_llm_calls_per_run: int = 300
    regression_margin: float = 0.10
    flaky_ci_width: float = 0.5
    temperature: float = 0.2
    db_path: str = "evalgate.db"
    costs: CostsConfig = field(default_factory=CostsConfig)
    edgar: EdgarConfig = field(default_factory=EdgarConfig)

    def validate(self) -> None:
        """Raise ValueError for any obviously wrong config values."""
        placeholder = "your-email@example.com"
        if placeholder in self.edgar.user_agent:
            raise ValueError(
                "Set [edgar] user_agent in evalgate.toml to your real email. "
                "SEC EDGAR requires a valid contact address in the User-Agent header."
            )
        if self.trials < 1 or self.trials > 20:
            raise ValueError("trials must be between 1 and 20")
        if self.edgar.mode not in ("live", "replay"):
            raise ValueError("edgar.mode must be 'live' or 'replay'")


def load_config(path: Path | None = None) -> EvalGateConfig:
    """Load evalgate.toml from `path` (defaults to CWD/evalgate.toml).

    Missing file → all defaults. Unrecognised keys are silently ignored so
    partial config files work without listing every field.
    """
    toml_path = path or Path("evalgate.toml")

    raw: dict = {}
    if toml_path.exists():
        with toml_path.open("rb") as f:
            raw = tomllib.load(f)

    eg = raw.get("evalgate", {})
    costs_raw = raw.get("costs", {})
    edgar_raw = raw.get("edgar", {})

    costs = CostsConfig(
        input_per_million=costs_raw.get("input_per_million", 0.075),
        output_per_million=costs_raw.get("output_per_million", 0.30),
    )
    edgar = EdgarConfig(
        user_agent=edgar_raw.get("user_agent", EdgarConfig.user_agent),
        mode=edgar_raw.get("mode", "replay"),
        fixtures_dir=edgar_raw.get("fixtures_dir", EdgarConfig.fixtures_dir),
        requests_per_second=edgar_raw.get("requests_per_second", 5),
    )

    return EvalGateConfig(
        trials=eg.get("trials", 6),
        max_concurrent_trials=eg.get("max_concurrent_trials", 4),
        trial_timeout_s=eg.get("trial_timeout_s", 120),
        judge_model=eg.get("judge_model", "gemini/gemini-2.0-flash"),
        max_llm_calls_per_run=eg.get("max_llm_calls_per_run", 300),
        regression_margin=eg.get("regression_margin", 0.10),
        flaky_ci_width=eg.get("flaky_ci_width", 0.5),
        temperature=eg.get("temperature", 0.2),
        db_path=eg.get("db_path", "evalgate.db"),
        costs=costs,
        edgar=edgar,
    )
