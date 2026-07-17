"""ScoreResult dataclass — the output shape every scorer produces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    """The outcome of one scorer applied to one trial."""

    scorer_type: str
    passed: bool
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorer_type": self.scorer_type,
            "passed": self.passed,
            "detail": self.detail,
            "extra": self.extra,
        }
