"""Pydantic v2 models for EvalCase, ScoringSpec, TrialResult, CaseResult, and RunReport."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# EvalCase input models (user-facing, validated on YAML load)
# ---------------------------------------------------------------------------


class NumericExpected(BaseModel):
    """Ground-truth value for the numeric scorer."""

    model_config = ConfigDict(extra="forbid")

    value: float
    tolerance_pct: float = 1.0

    @field_validator("tolerance_pct")
    @classmethod
    def positive_tolerance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("tolerance_pct must be > 0")
        return v


class TrajectoryStep(BaseModel):
    """One expected tool call in the trajectory scorer."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    # Partial key→value match on the tool's args. Each value is compared as str().
    args_contain: dict[str, Any] = Field(default_factory=dict)


class Expected(BaseModel):
    """All optional ground-truth blocks. A case may use any subset."""

    model_config = ConfigDict(extra="forbid")

    numeric: NumericExpected | None = None
    # Every string in this list must appear in final_text (case-insensitive).
    contains: list[str] = Field(default_factory=list)
    # Ordered subsequence of expected tool calls.
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    # Full-text regex applied to final_text.
    regex: str | None = None


class ScoringEntry(BaseModel):
    """One scorer to apply to each trial."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["numeric", "contains", "regex", "trajectory", "judge"]
    # Required when type="judge".
    rubric: str | None = None
    # When True, trajectory scorer requires exact match instead of subsequence.
    exact: bool = False


class Thresholds(BaseModel):
    """Pass/fail thresholds at the case level."""

    model_config = ConfigDict(extra="forbid")

    pass_rate_min: float = 0.75


class EvalCase(BaseModel):
    """A single eval case as loaded from a YAML file."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    input: str
    tags: list[str] = Field(default_factory=list)
    # Overrides config default when present. Range: 1–20.
    trials: int = 6
    expected: Expected = Field(default_factory=Expected)
    scoring: list[ScoringEntry]
    pass_policy: Literal["all", "any"] = "all"
    thresholds: Thresholds = Field(default_factory=Thresholds)

    @field_validator("id")
    @classmethod
    def valid_id_chars(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError(
                f"id {v!r} must contain only lowercase letters, digits, and underscores"
            )
        return v

    @field_validator("trials")
    @classmethod
    def trials_in_range(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError(f"trials must be between 1 and 20, got {v}")
        return v

    @field_validator("scoring")
    @classmethod
    def at_least_one_scorer(cls, v: list[ScoringEntry]) -> list[ScoringEntry]:
        if not v:
            raise ValueError("at least one scoring entry is required")
        return v

    @model_validator(mode="after")
    def scoring_matches_expected(self) -> EvalCase:
        """Each scorer type must have its matching expected block."""
        for entry in self.scoring:
            if entry.type == "numeric" and self.expected.numeric is None:
                raise ValueError(
                    "scoring type 'numeric' requires an expected.numeric block "
                    "(add 'numeric: {value: ..., tolerance_pct: ...}' under 'expected:')"
                )
            if entry.type == "trajectory" and not self.expected.trajectory:
                raise ValueError(
                    "scoring type 'trajectory' requires a non-empty expected.trajectory list"
                )
            if entry.type == "judge" and not entry.rubric:
                raise ValueError(
                    "scoring type 'judge' requires a rubric string in the scoring entry"
                )
            if entry.type == "regex" and self.expected.regex is None:
                raise ValueError("scoring type 'regex' requires an expected.regex string")
            if entry.type == "contains" and not self.expected.contains:
                raise ValueError(
                    "scoring type 'contains' requires a non-empty expected.contains list"
                )
        return self


# ---------------------------------------------------------------------------
# Result types (internal, populated by runner and scorers)
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Everything produced by one trial (one call to adapter.run + scoring)."""

    trial_idx: int
    passed: bool
    failure_reason: str | None = None
    final_text: str = ""
    # list[ScoreResult] serialised to dicts via ScoreResult.to_dict().
    scores: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    raw_events: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # Judge prompt/response for the (single) judge scorer, kept at trial level
    # because Phase 7 calibration reads them from dedicated DB columns.
    judge_prompt: str | None = None
    judge_response: str | None = None


@dataclass
class CaseResult:
    """Aggregated result for one EvalCase across all its trials."""

    case_id: str
    trials: list[TrialResult]

    @property
    def passes(self) -> int:
        return sum(1 for t in self.trials if t.passed)

    @property
    def pass_rate(self) -> float:
        n = len(self.trials)
        return self.passes / n if n else 0.0

    @property
    def latencies_ms(self) -> list[float]:
        return [t.latency_ms for t in self.trials if t.latency_ms > 0]

    @property
    def p50_latency_ms(self) -> float:
        lats = self.latencies_ms
        return statistics.median(lats) if lats else 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.trials)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.trials)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.trials)


@dataclass
class RunReport:
    """Top-level result for one evalgate run (all cases)."""

    cases: list[CaseResult]
    agent_name: str = "sec_agent"

    @property
    def total_cost_usd(self) -> float:
        return sum(c.total_cost_usd for c in self.cases)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.total_input_tokens for c in self.cases)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.total_output_tokens for c in self.cases)
