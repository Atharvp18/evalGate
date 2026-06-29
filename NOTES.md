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

---

## 2026-06-29 — Phase 1: EDGAR Client + Fixtures

**Key decisions:**

- **Two separate trim functions (`_trim_facts_for_storage` vs `_trim_facts_for_context`).** Storage
  trim discards unknown XBRL concepts but keeps the full historical series for the ones we care about,
  so fixtures remain reusable if we add concepts later. Context trim then downsamples to ≤12 entries
  per concept so the LLM receives ~8 KB instead of a raw 10 MB dump. Conflating them would either
  bloat fixtures or silently discard history.

- **No `httpx.Client` constructed in replay mode.** The spec says replay must never touch the network.
  Enforced structurally: `self._http` is never created in replay mode, so any accidental call would
  raise `AttributeError` immediately rather than silently hitting the network. A runtime `if mode ==
  "replay": raise` guard inside `_get()` would be easier to accidentally remove.

- **Fixture filenames are `sha256(url)[:16].json`.** Collision-proof, stable across re-runs, and
  short enough to be git-friendly. `index.json` alongside maps URLs → filenames so a human can open
  it and see what each file contains without decoding hashes.

- **Context-budget trimming lives in the tool function, not the client.** `get_company_facts()` calls
  `_trim_facts_for_context()` on whatever the client returns. The client stays a pure fetch/replay
  layer. This keeps a clean `get_company_facts_raw()` escape hatch for debugging without touching
  the client.

- **Module-level singleton (`configure_client` / `_require_client`).** ADK tool functions are
  registered as callables with no extra arguments — the framework calls them with only the LLM-
  supplied args. Passing the client as an explicit argument is not possible in that calling
  convention, so a module-level singleton is the right pattern here.

**Bugs encountered:**

- `record_fixtures.py` failed with `ModuleNotFoundError: No module named 'examples'` because the
  `sys.path.insert` only added `src/` but not the repo root. Root cause: the script runs from
  `examples/sec_agent/`, so Python adds that directory to `sys.path`, not the repo root. Fix: also
  insert `_repo_root` (two levels up from the script) into `sys.path`.

**Surprises:**

- Nvidia's XBRL facts JSON contains both `Revenues` and period-level aggregate entries under the
  same concept. Some entries have `form="10-K"` with much larger values (cumulative annual) next to
  the quarterly `10-Q` entries. The context trim filters to `10-Q` and `10-K` only (dropping `8-K`,
  amendments, etc.), but the user must be aware that `10-K` values are annual totals, not quarterly.

- Raw fixture sizes before trimming: largest was ~8 MB (Apple XBRL). After storage trim to known
  concepts only, all fixtures are well under 3 MB. No manual intervention was needed.

- Nvidia's most recent quarterly revenue (Q1 FY2027, period ending 2026-04-26): **$81.6 billion**.
