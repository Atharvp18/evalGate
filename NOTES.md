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

---

## 2026-07-15 — Phase 3: Schema, Loader, Async Runner

**Key decisions:**

- **Pydantic v2 `extra="forbid"` on every model.** Typos in YAML keys silently pass when using
  dicts; Pydantic turns them into immediate, actionable errors pointing to the field name. A user
  writing `rubrics:` instead of `rubric:` gets a message saying "Extra inputs are not permitted"
  rather than a silent wrong-scorer configuration.

- **`@model_validator(mode="after")` for cross-field scoring constraints.** The rule "if type is
  'numeric', expected.numeric must be present" cannot be expressed as a single-field validator
  because it reads two fields. The after-mode model validator runs after all fields are parsed and
  typed, so `self.expected.numeric` is already a `NumericExpected | None`, not a raw dict. This
  is the correct place for invariants that span multiple fields.

- **Per-trial exception catch + continue semantics in runner.** Any exception inside `_run_single_trial`
  is caught and stored as `failure_reason=repr(e)` in the TrialResult. The run never crashes because
  one trial throws. This matters on the free tier: transient Gemini 429s or network blips would abort
  a 6-case × 8-trial run without this guard. The spec explicitly requires this: "Any unexpected
  exception in a trial is captured as a failed trial... the run as a whole must never die because one
  trial threw."

- **`asyncio.Semaphore` scoped across the whole run, not per-case.** All cases' trials compete for
  the same pool of `max_concurrent_trials` slots. This means a large case can't monopolize concurrency
  while a smaller case waits — all trials across all cases are scheduled fairly by asyncio's event
  loop. The semaphore is created once in `run_cases()` and passed down to each `_run_single_trial`.

- **Single-element list for the LLM call counter** (`llm_call_counter = [0]`). Python integers are
  immutable; a closure over `count += 1` inside a coroutine only mutates the local binding. A
  single-element list gives a mutable container that all coroutines on the same event loop share
  without threading primitives. Alternative is `nonlocal` but that requires nesting; the list pattern
  is more readable and common in asyncio code.

- **Phase 3 placeholder scorer: always pass.** The runner sets `passed=True` for every trial that
  doesn't time out or raise. This is correct by the phase contract: "trials with scores empty and a
  temporary always-pass placeholder so the loop is observable end to end." Phase 4 replaces this with
  real scorer output.

- **Result types as dataclasses, not Pydantic models.** `TrialResult` and `CaseResult` are internal
  to the framework — they are never user-facing YAML. Pydantic's validation overhead and the
  `extra="forbid"` discipline are only needed for user input. Dataclasses are lighter and equally
  type-safe for internal pipeline types.

**Bugs encountered:** None in this phase.

**Surprises:**

- The `statistics.quantiles(data, n=20)[18]` approach for p95 requires at least 2 data points.
  Added a guard for runs with a single trial.

- Ruff's B904 rule flags `raise typer.Exit()` inside `except` blocks because it looks like you
  might be losing the original exception. But `typer.Exit` is a control-flow signal, not an error
  — the user already saw the error message on the line above. Resolved with `raise ... from None`
  which explicitly says "I am intentionally not chaining the original exception."

---

## 2026-07-17 — Phase 4: Scorers

**Key decisions:**

- **Number normalizer as one regex + a scale-word table.** Candidates are extracted with a single
  regex that matches comma-grouped or plain numbers with an optional `$` prefix and an optional
  scale suffix (`thousand|million|billion|trillion|k|m|b|t|mn|bn`). Numbers immediately followed
  by `%` are skipped — percentages are almost never the dollar answer being checked. Regex
  backtracking naturally prevents false suffixes: in "300 barrels", the `b` of "barrels" fails the
  trailing `\b`, so the match falls back to plain `300`.

- **Trajectory matching is subsequence by default, exact opt-in.** Agents legitimately take extra
  steps (sub-agent transfers, retries, extra lookups) — punishing them for that creates false
  failures. What matters is the required calls happened in the required order. `exact: true` is
  available for cases where any extra call is itself the bug.

- **Judge never sees expected answers — rubric only.** If the judge saw the expected value it would
  become a noisy re-implementation of the numeric scorer, and Phase 7 calibration against human
  labels would be meaningless. The prompt is rubric + question + agent answer, demanding strict
  JSON `{"pass": bool, "reason": str}`.

- **Judge parsing: strip code fences, validate `pass` is a real bool, retry once, then fail.**
  Gemini often wraps JSON in ```` ```json ```` fences. `_parse_judge_json` also rejects
  `"pass": "yes"` (string, not bool) — a silent truthy-string bug otherwise. After one retry the
  score is `passed=False, detail="judge_output_unparseable"` rather than a crashed trial.

- **Scorer dispatch is a plain if/elif in `score_trial()`, not a registry.** Five fixed scorer
  types known at schema level (a `Literal`) do not need a plugin registry; the spec's
  "no premature abstraction" rule applies. Deterministic scorers are sync functions; only the
  judge is async — one async dispatch function is simpler than forcing a uniform async Protocol.

- **Scoring exceptions are a failed trial, not a crashed run.** The runner wraps `score_trial` in
  its own try/except (same contract as adapter errors) — a judge network blip fails one trial
  with the traceback stored and the run continues.

- **`judge_prompt`/`judge_response` lifted to TrialResult fields.** The Phase 5 SQLite schema has
  dedicated columns for them (calibration reads them in Phase 7), so the runner copies them out
  of the judge's ScoreResult.extra into the trial record.

**Bugs encountered:** None in this phase.

**Surprises:**

- The percentage-skip check must look at the text *after* `lstrip()` — "12 %" with a space is a
  thing in model output.

- `litellm` is imported lazily inside `score_judge` because its import is slow (~1s) and pulls in
  a large dependency tree; unit tests and judge-free runs never pay that cost.

---

## 2026-07-18 — Phase 4 addendum: free-tier reality check

**Bugs encountered (after first live scored run):**

- **All trials 429'd with `RESOURCE_EXHAUSTED`.** Two separate Gemini free-tier quotas were hit:
  first the per-minute cap (5 req/min — one trial alone makes ~4 model calls, and 4 concurrent
  trials blew it instantly), then the per-day cap (20 req/day for `gemini-2.5-flash`), which no
  retry can beat. Fixes: `max_concurrent_trials` 4 → 1, one 60s-backoff retry on rate-limit
  tracebacks in the runner, `num_retries=2` on judge calls, and — the real fix — switching agent
  and judge to `gemini-3.1-flash-lite`, which has a far larger free daily quota.
  (`gemini-2.5-flash-lite` 404s: "no longer available to new users".)

- **Judge flagged correct answers as "invented future dates".** The judge model's training data
  predates the 2026 filing dates in the fixtures, so it ruled them fabricated. Fix: a prompt line
  telling the judge to judge only against the rubric and never use its own knowledge of dates or
  figures. Lesson: an LLM judge silently imports its own world model unless explicitly fenced.

**Open finding:** `nvda_aapl_comparison` fails consistently (0/3 trials) — the judge reports the
agent uses outdated or wrong-concept Apple data for year-over-year growth. Single-company Apple
retrieval passes, so the suspect is the fixture trim not retaining year-ago quarters. To
investigate; this is the framework catching a real agent defect.

---

## 2026-07-19 — Phase 5: Stats, Store, Report

**Key decisions:**

- **Wilson interval implemented directly; scipy only supplies the z-value.** The formula is six
  lines. Wilson beats the naive p̂ ± 1.96·SE interval in exactly our regime (small N, extreme
  rates): naive at 5/5 gives a zero-width interval — false certainty from five samples — and can
  exceed [0, 1]. Wilson pulls the centre toward 0.5 and always stays in bounds. Unit-tested
  against hand-computed values (8/10 → (0.490, 0.943)).

- **Flaky requires BOTH partial passes AND CI width > threshold.** 15/20 passes has a CI width of
  ~0.36 — mostly reliable, not flaky. 2/4 has width ~0.70 — genuinely too noisy to trust either
  verdict. Flaky is a distinct third verdict so a human investigates instead of trusting pass or
  fail.

- **`load_run` returns trial rows under `"trial_rows"`, not `"trials"`.** The case_results row
  already has a `trials` column (the count); reusing the key would silently clobber it in the
  merged dict. Found while writing the round-trip test — the test asserted both the count and the
  rows through the same key.

- **One transaction per run (`with conn:`).** A crash mid-save leaves the DB with no partial run
  rather than a run with half its trials. sqlite3's context manager commits on success, rolls
  back on exception — no explicit BEGIN/COMMIT needed.

- **`config_json` serialised with `default=lambda o: o.__dict__`.** EvalGateConfig is nested
  dataclasses; this one-liner flattens them without pulling in a serialisation library.

**Bugs encountered:**

- **p50 > p95 in the report summary.** Hand-rolled percentile indexing (`lats[int(len*0.95)-1]`)
  returns index 0 — the *minimum* — for a 2-element list. Spotted because the printed p50
  (4595 ms) exceeded p95 (2960 ms), which is impossible. Fix: `statistics.median` /
  `statistics.quantiles`, same as the run summary already used. Lesson: don't hand-roll
  percentile math when stdlib has it.

**Surprises:**

- sqlite3's `conn.executescript` cannot run inside a transaction, but running it on every
  `connect()` is idempotent thanks to `IF NOT EXISTS` — no separate migration step needed at
  this scale.
