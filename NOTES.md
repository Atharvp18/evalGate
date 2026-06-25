# EvalGate — Engineering Journal

Append-only. After every phase: date, phase, key decisions, bugs + root cause, surprises.
Used as interview prep — keep entries factual and specific.

---

## 2026-06-25 — Phase 0: Scaffold

**Key decisions:**

- **`tomllib` for TOML parsing (stdlib, Python 3.11+).** No extra dep needed. We require Python 3.12+
  so this is always available. Alternative was `tomli` (third-party backport) but stdlib is simpler.

- **`hatchling` as build backend.** Lightweight, well-supported by `uv`, no config beyond
  `[tool.hatch.build.targets.wheel] packages = ["src/evalgate"]`. Alternative was `setuptools` but
  hatchling's src-layout support is cleaner out of the box.

- **Dataclasses (not Pydantic) for `EvalGateConfig`.** Config is internal, loaded once at startup,
  not user-visible schema. Pydantic v2 is reserved for the user-facing `EvalCase` YAML schema where
  the validation error messages matter. Dataclasses are lighter here.

- **Placeholder email check in `validate()`.** SEC EDGAR requires a valid contact address in the
  `User-Agent` header. Failing loudly at startup (before any network call) is better than getting
  silently blocked by EDGAR after running 10 trials.

- **`ruff` for both lint and format.** Replaces black + flake8 + isort in one tool. Configured in
  `pyproject.toml` with `select = ["E", "F", "I", "UP", "B", "SIM"]`. `B008` (do not perform function
  calls in argument defaults — Typer needs this for `typer.Option(...)`) is suppressed.

**Bugs encountered:** None in this phase.

**Surprises:** `uv` was already installed at `/opt/homebrew/bin/uv` but not on `$PATH` in the Claude
Code shell. Confirmed by using the full path. User should add `/opt/homebrew/bin` to their `$PATH`.
