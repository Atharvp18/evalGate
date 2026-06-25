"""Tests for config loading — Phase 0."""

from pathlib import Path

import pytest

from evalgate.config import EvalGateConfig, load_config


def test_load_defaults_when_no_file(tmp_path: Path) -> None:
    """load_config with a non-existent path returns all default values."""
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert isinstance(cfg, EvalGateConfig)
    assert cfg.trials == 6
    assert cfg.max_concurrent_trials == 4
    assert cfg.trial_timeout_s == 120
    assert cfg.max_llm_calls_per_run == 300
    assert cfg.regression_margin == pytest.approx(0.10)
    assert cfg.flaky_ci_width == pytest.approx(0.5)


def test_load_overrides_from_toml(tmp_path: Path) -> None:
    """Values in evalgate.toml override defaults."""
    toml = tmp_path / "evalgate.toml"
    toml.write_text("[evalgate]\ntrials = 10\nmax_concurrent_trials = 2\n")
    cfg = load_config(toml)
    assert cfg.trials == 10
    assert cfg.max_concurrent_trials == 2
    # unspecified fields keep defaults
    assert cfg.trial_timeout_s == 120


def test_validate_raises_on_placeholder_email(tmp_path: Path) -> None:
    """validate() raises ValueError when edgar.user_agent is still the placeholder."""
    cfg = load_config(tmp_path / "nonexistent.toml")
    with pytest.raises(ValueError, match="user_agent"):
        cfg.validate()


def test_validate_passes_with_real_email(tmp_path: Path) -> None:
    """validate() succeeds when edgar.user_agent has a real email."""
    toml = tmp_path / "evalgate.toml"
    toml.write_text('[edgar]\nuser_agent = "EvalGate <real@example.com>"\n')
    cfg = load_config(toml)
    cfg.validate()  # should not raise
