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

---

## 2026-06-29 — Phase 2: Subject Agent + ADKAdapter

**Key decisions:**

- **`AgentAdapter` is a `Protocol`, not an ABC.** Structural typing means future
  adapters do not need to inherit from anything — they just implement
  `async def run(self, query: str) -> AgentRunResult`. This matches the spec's
  "no premature abstraction" rule: we have exactly one implementation, and the
  Protocol exists only so the runner can type-check against it.

- **Fresh session per `adapter.run()` call.** ADK's `InMemorySessionService`
  keys sessions by `(app_name, user_id, session_id)`. We generate a new
  `uuid.uuid4()` session id every call, so trials cannot leak state to each
  other. This is a correctness requirement: shared session memory would
  correlate trials, invalidating the Wilson CI assumption that trials are
  independent samples.

- **Module-level singleton for EDGAR client (`configure_client`).** ADK tool
  functions are passed by reference into `LlmAgent(tools=[...])`. ADK then
  invokes them with only the args the LLM emits — there is no way to inject
  extra arguments like an `EdgarClient`. A module-level singleton, configured
  at agent startup, is the cleanest pattern that fits ADK's calling convention.

- **Safe calculator via `ast.parse(mode='eval')` + a whitelist.** Built around
  a small dict of allowed `BinOp` / `UnaryOp` node types. Rejects names, calls,
  attribute access, subscripts, comprehensions — anything that could escape
  the arithmetic domain. Tested with `__import__('os').system(...)` as a
  smoke test for the rejection path.

- **Token usage summed across every event.** ADK emits multiple events per
  invocation when sub-agents are involved (one per LLM call per sub-agent
  hop). The adapter sums `prompt_token_count + candidates_token_count` across
  all of them — a single trial of one query consumed 17,433 input tokens
  during verification (coordinator + retrieval + report each call the model
  with the full context, hence the high count).

- **`raw_events` stored as a lightweight dict snapshot, not the full Event.**
  Full ADK Events contain internal references that do not JSON-serialise
  cleanly. The adapter extracts `{author, is_final, text, function_calls,
  function_responses}` per event — enough for `mine-trace` (Phase 8) to
  reconstruct what happened.

- **`gemini-2.5-flash` over `gemini-2.0-flash`.** The 2.0 model returned
  `quota: 0` on the AI Studio free tier (regional restriction); 2.5-flash
  worked immediately. Updated `evalgate.toml` and `config.py` defaults.

**Bugs encountered:**

- API key with prefix `AQ.Ab8RN...` got `RESOURCE_EXHAUSTED limit: 0` for
  `gemini-2.0-flash`. Same key worked fine for `gemini-2.5-flash`. Root cause:
  not all models are available on the free tier in every region. The model
  list call (`client.models.list()`) succeeded with the same key, so the key
  itself was valid — it was a per-model quota issue. The new key prefix `AQ.`
  is the standard format now (older `AIza` prefix is deprecated).

- `ruff` flagged `E402` on `chat.py` and `record_fixtures.py` because those
  scripts manipulate `sys.path` before importing. Resolved with a per-file
  ignore in `pyproject.toml` rather than restructuring — these are entry-point
  scripts that need to bootstrap the package before installation.

**Surprises:**

- Even for a simple lookup ("Nvidia's revenue last quarter"), the coordinator
  made 4 sub-agent transfers and tool calls: `transfer_to_agent(retrieval)`
  → `lookup_cik` → `get_company_facts` → `transfer_to_agent(report)`.
  Analysis was correctly skipped because no math was needed — the instruction
  to "skip step 2 if the question can be answered without analysis" worked
  on the first try.

- First live run latency: ~21 seconds. Three sequential LLM calls (coordinator
  decides → retrieval executes → report formats), each going to a free-tier
  endpoint. This will matter for eval runs: 6 cases × 8 trials × 21 s = ~17
  minutes per `evalgate run`. Phase 3 needs the asyncio semaphore precisely
  because of this.

- The agent cited the exact period_end date and form type without being asked
  to in the question. The report agent's instruction
  ("Use the company name and the source filing for every figure. Always
  cite.") propagated correctly through the coordinator's hand-off.
