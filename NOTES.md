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

---

## 2026-07-20 — Phase 6: pytest plugin + compare/gate logic

**Key decisions:**

- **Regression requires BOTH a margin breach AND CI exclusion.** A case only regresses when
  `current.pass_rate < baseline.pass_rate - regression_margin` (default 0.10) AND the current
  run's Wilson CI does not contain the baseline's pass rate. Verified the boundary directly:
  6/8 → 5/8 clears the margin (drop of 0.125) but the 5/8 CI (≈[0.31, 0.86]) still contains
  0.75, so it stays "ok" — that drop is statistically indistinguishable from noise at n=8. Only
  8/8 → 1/8 (both conditions true) flags "regressed". Margin alone would false-alarm on every
  noisy small-N wobble; CI alone would flag rounding-error-sized drops at large N.

- **New cases never gate; removed cases only warn.** Both get informational verdicts excluded
  from `has_regressions()`. A case with no baseline has nothing to regress against — gating on
  it would punish the PR that added coverage.

- **pytest plugin runs the engine once per session, cached on `session.stash`.** Same principle
  as the Phase 3 semaphore: don't multiply LLM cost by something that isn't inherent to the
  work. The first pytest item to need results triggers `run_cases()` once; every other item
  reads its own `CaseResult` out of the cached dict.

- **`load_case_file()` factored out of `load_cases()`.** The plugin validates one YAML file per
  collected item (for per-file error messages); `evalgate run` validates a whole directory with
  duplicate-id checking. Both now share one parse/validate path instead of two copies that could
  drift.

**Bugs encountered:**

- **`pytest_load_initial_conftests` silently failed to inject the cases directory.** First
  version guarded with `if cases_dir not in args: args.append(cases_dir)` — but `cases_dir`
  already appears in `args` as the `--evalgate` option's *value*, so the guard always saw a
  "duplicate" and skipped the append. Result: `pytest --evalgate DIR` fell back to `testpaths`
  and only ever collected `tests/`, never the YAML cases. Caught by instrumenting the hook with
  a debug print and comparing `args` before/after — fixed by always appending (pytest tolerates
  the same path appearing as both an option value and a positional collection root).

**Surprises:**

- Verifying `compare` end-to-end against the real agent (2-case subset, replay mode) worked
  cleanly, but the full 15-case suite still hits the free-tier `RESOURCE_EXHAUSTED` rate limit
  from the Phase 4 addendum (15 req/min on `gemini-3.1-flash-lite`) — 15 cases × N trials queues
  up 60-second backoff retries fast. Not a new bug, just a reminder that day-to-day dev-loop runs
  against the real agent should use a small case subset, not the full suite.

---

## 2026-07-21 — Phase 7: Judge Calibration

**Key decisions:**

- **The original question is recovered from the stored `judge_prompt`, not a new DB column.**
  `JUDGE_PROMPT_TEMPLATE` (scorers/judge.py) embeds the question between two fixed string
  markers (`"Question given to the agent:\n"` … `"\n\nAgent's final answer:"`), and the full
  prompt is already persisted verbatim in `trials.judge_prompt`. Splitting on those markers
  round-trips the question with zero schema change. Falls back to the raw prompt text if the
  markers are ever missing (e.g. the template changes later) rather than raising — labelling
  can proceed with slightly noisier context instead of a hard failure.

- **Stratified sampling is round-robin across `case_id`, not random.** A case with many
  judge-scored trials (e.g. a flaky one re-run often) cannot crowd out a case with only one
  or two — every case gets a pick before any case gets a second one. Simpler than weighted
  random sampling and deterministic, which matters for reproducible exports.

- **Cohen's kappa over raw agreement as the headline number.** Raw agreement is inflated when
  one class dominates — a judge that always says "pass" agrees with a mostly-passing human
  90% of the time by chance alone. Kappa subtracts expected chance agreement, so it is the
  number that actually says whether the judge adds signal. Both are reported, but kappa is
  what should be quoted in the README/resume per the spec.

- **`--export` and the compute path share one `--labels` path, not two flags.** The command
  writes to `--labels` when `--export` is set and reads from the same path otherwise. One flag,
  one file, two directions — matches the spec's `evalgate calibrate --labels ... [--export]`
  contract and avoids a second path argument that could silently point at the wrong file.

**Bugs encountered:**

- **First draft returned the judge's own JSON verdict as `final_text` in the export CSV**, not
  the agent's actual answer. `sample_judge_trials` selected `t.judge_response` (the judge
  model's `{"pass": ..., "reason": ...}` reply) into the `final_text` field instead of the
  trial's own `t.final_text` column. Invisible in the unit tests because the test fixture set
  both fields to the same string. Caught by smoke-testing `--export` against the real
  `evalgate.db` and eyeballing the CSV — the `final_text` column was literal judge JSON. Fixed
  by selecting `t.final_text` in the query; added an assertion in `test_export_and_load_round_trip`
  that `final_text` doesn't contain `"pass"` to prevent silent regression.

**Surprises:**

- The real DB only has 2 judge-scored trials right now (one case, `msft_revenue_contains`, run
  twice) — nowhere near the ~100 the spec wants for a meaningful kappa. Getting there requires
  running the full 15-case suite (8 of which use the judge scorer) enough times to accumulate
  ~100 judge trials, which the Phase 6 notes already flagged as rate-limit-bound on the free
  tier. Calibration code is done and smoke-tested end-to-end (export → hand-label → compute →
  report), but the actual ~100-label human pass and the real kappa number are still open —
  that's the next piece of legwork before this phase can close out.

---

## 2026-07-22 — Phase 8: CI Gate, Trace Mining, Dashboard

**Key decisions:**

- **`mine-trace` recovers the input question from the case's own YAML file, not from a DB
  column.** `TrialResult` only ever stored the agent's output (`final_text`), never its input —
  every trial for a case is run against the same fixed `input` string in that case's YAML, so
  re-loading the case by `case_id` via `loader.load_cases()` is the exact original text, not a
  reconstruction. Mirrors the Phase 7 decision to recover data from an existing source of truth
  instead of adding a schema column for something already available elsewhere.

- **Mined trajectories drop `transfer_to_agent` calls.** Those are ADK sub-agent hand-off
  plumbing, not something a human would ever assert on in a trajectory scorer — keeping them
  would make every mined case's trajectory scorer immediately fail on the next passing run just
  from ordinary agent routing.

- **The judge rubric is always a `TODO` placeholder, never inferred from the failure.** Writing
  a rubric requires human judgment about what "correct" looks like; auto-generating one from a
  failed trial's own defect would just encode that defect as the new "expected" behavior. The
  trajectory block only appears when the trial actually called at least one real tool — a
  transfer-only trial mines to a judge-only draft.

- **The nightly live-smoke check is a small script (`examples/sec_agent/live_smoke.py`), not
  inline Python in the workflow YAML.** A multi-line Python block embedded in `run:` is unreadable
  in the Actions UI and impossible to lint or unit-test; a real script file gets both, and can be
  run locally (`python examples/sec_agent/live_smoke.py`) to debug a red nightly run without
  touching CI at all.

- **`baselines/evalgate.db` (committed to git) is restored by copying it to the working DB path
  before `evalgate run`, per the spec's explicit "simplest correct approach" call.** The
  alternative — passing the baseline between jobs as a GitHub Actions artifact — avoids
  committing a binary file to git, but couples the gate to artifact retention windows and adds a
  second job. A baseline is a single SQLite file that changes rarely (only when a PR intentionally
  updates it); committing it keeps the gate a two-step, single-job workflow with no external state.

- **Dashboard reads SQLite directly into pandas with hand-written SQL joins, no ORM and no new
  `store.py` functions.** `pandas.read_sql_query` against `connect()` needs exactly three ad-hoc
  joins (case_results×runs, trials×case_results) that only the dashboard uses — adding them to
  `store.py` as named functions would be an abstraction with a single caller. `st.cache_data(ttl=30)`
  avoids re-opening the DB on every widget interaction without needing a manual refresh button.

**Bugs encountered:**

- **`yaml.safe_dump` escaped the em dash in the mined description as `—`** because
  `allow_unicode` defaults to `False` — the header comment and description both use "—" for
  readability elsewhere in the project. Fixed with `allow_unicode=True`; caught immediately by
  smoke-testing `mine-trace` against the real `evalgate.db` and reading the output file, same as
  the Phase 7 `final_text` bug — unit tests parse the YAML back with `yaml.safe_load` either way,
  so they never would have caught the escaping.

**Surprises:**

- `evalgate mine-trace` works on a *passing* trial too (nothing in the code requires
  `passed=False`) — smoke-tested against real trial 1 in `evalgate.db`, which actually passed.
  The spec's "failed trial" framing is the intended use case, not an enforced precondition; there
  was no reason to add a check that would only reject a harmless call.

**Still open (requires real API calls / real PRs, deferred rather than done silently):**

- The nightly-smoke script was syntax-checked and YAML-validated but not executed — running it
  spends live Gemini + SEC EDGAR quota.
- The "real PR blocked, screenshot in README" deliverable from the spec's Phase 8 "Done when"
  needs an actual GitHub PR now that the baseline exists — not something to fabricate.

---

## 2026-07-22 — Phase 8 addendum: seeding the baseline hit the daily quota, again

**Bugs encountered:**

- Even `gemini-3.1-flash-lite` free tier only allows **15 requests/minute** per the 429 response
  body (`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, quotaValue 15) — a full 15-case ×
  3-4-trial run needs 200+ calls (agent-internal coordinator/retrieval/report hops plus judge
  calls on 8 cases), which cannot fit inside that window even serialized
  (`max_concurrent_trials=1`) and with the runner's one-retry-after-60s backoff. Confirmed this is
  a per-minute cap, not the previously-hit per-day cap from the Phase 4 addendum.

- **Fix for *seeding* the baseline (not a permanent config change): `evalgate run --trials 1`.**
  Overriding every case to 1 trial cut the call volume enough (15 cases × ~1 call each, judge
  calls only on judge-scored cases) that only 2 of 15 cases still hit the per-minute window and
  failed as a result — `flaky_ticker_only` (crashed mid-retry) and `googl_amzn_comparison`
  (scoring-stage litellm exception). Both are quota artifacts, not real agent defects.

- **`nvda_aapl_comparison` failed too, but this is the pre-existing real defect** first logged in
  the Phase 4 addendum (Apple year-over-year growth using wrong-concept/outdated data) — the
  judge correctly flagged "inconsistent and illogical fiscal periods." Not a new bug; confirms the
  open finding is still live and unfixed.

**Key decision:** saved run 3 (the `--trials 1` run, 12/15 clean passes) as the `main` baseline
as-is, including the 2 quota-artifact failures and the 1 real defect, rather than spending more
quota to retry just the 2 artifacts. Consequence to remember: those 3 cases currently show 0% in
the baseline. A future run that also fails them will look "unchanged" (correct), but a future run
that fixes the real `nvda_aapl_comparison` defect will show as "improved" rather than confirming a
regression was fixed — acceptable for now, but this baseline should be re-seeded with a full,
clean, multi-trial run once quota allows, rather than trusted long-term as-is.

**Also found:** `.gitignore`'s bare `evalgate.db` pattern was silently excluding
`baselines/evalgate.db` too (gitignore patterns without a leading `/` match the basename at any
depth). Spec explicitly requires the baseline DB committed. Fixed with a negation line
(`!baselines/evalgate.db`) directly under the exclusion rule so the exception stays next to what
it's excepting.

---

## 2026-07-22 — Phase 9: Regression-Injection Study (code built, live run pending)

**Key decisions:**

- **Injections are in-process monkeypatches (`unittest.mock.patch`), not on-disk file patches.**
  The spec says "via temporary patches" without mandating the mechanism. Monkeypatching a module
  attribute for the duration of a `with` block guarantees the original is restored even if the
  study crashes mid-injection — an on-disk patch-and-revert would need its own crash-safe cleanup
  to get the same guarantee, and a crash there could leave a real source file (or worse, a
  committed fixture) modified. In-process patching also means the whole study runs without ever
  writing to the working tree, so `git status` stays clean throughout.

- **`build_agent()` gained `temperature`, `retrieval_tools`, and `analysis_tools` parameters —
  and this surfaced a real latent bug.** `config.temperature` (default 0.2) was defined in
  `config.py` since Phase 0 but was never actually passed to the agent anywhere — `build_agent()`
  didn't accept it and no `generate_content_config` was ever set on the ADK `LlmAgent`s, so every
  run (eval, REPL, `adk run`) silently used ADK's own default temperature regardless of what
  `evalgate.toml` said. Wiring `generate_content_config=GenerateContentConfig(temperature=...)`
  into all four `LlmAgent`s fixes this for every caller, not just the study — `cli.py`'s
  `_build_sec_agent_adapter` and `chat.py` were both updated to pass `cfg.temperature` through.
  The tool-list parameters (`retrieval_tools`/`analysis_tools`) exist purely for injections #4, #6,
  #7, which need a different tool list than the hardcoded literal in `build_agent()`'s body —
  `None` (the default) preserves the exact original tool lists for every other caller.

- **Injections that only need a different tool list or temperature are expressed as
  `build_agent()` keyword overrides, not patches.** Only injections that touch something with no
  existing seam (prompts, the trim function, the EDGAR client's raw response) use
  `mock.patch.object`. This keeps the "patch" mechanism reserved for things that genuinely have no
  cleaner expression, rather than monkeypatching everything uniformly for consistency's own sake.

- **Naive mode strips scoring down to `{contains, regex, numeric}` and forces `trials=1`, dropping
  `judge` and `trajectory` entirely** — not because those scorers are hard to naive-ify, but
  because a team without an eval framework is unlikely to have built an LLM judge or a tool-call
  sequence checker in the first place. All 15 existing cases keep at least one deterministic
  scorer after the filter (verified this before writing the code, not after), so no case needs
  special-casing.

- **Naive mode's regression rule is a bare pass-to-fail flip per case, with no margin and no CI** —
  that absence is exactly what naive mode is a strawman for. `compare.py`'s real two-condition
  gate (margin AND CI exclusion) is reused unmodified for full mode, computed from `CaseResult` via
  `stats.wilson_interval` directly rather than round-tripping through SQLite — the study needs the
  same math `store.py` writes to the DB, not the DB itself.

- **The "before" (unmodified-agent) run is computed once per mode and shared across all 10
  injections, not re-run per injection.** Re-running the baseline 10 times would double the cost
  for no signal — the agent under test doesn't change between injections, only the record of what
  it does without any injection applied.

**Bugs encountered:**

- **`_build_adapter(build_kwargs, cfg)` crashed with `got multiple values for keyword argument
  'temperature'` on injection #5 specifically.** The function called
  `build_agent(temperature=cfg.temperature, **build_kwargs)`, and injection #5's own
  `build_kwargs = {"temperature": 1.0}` collided with the explicit keyword. Every other injection's
  `build_kwargs` doesn't touch `temperature`, so this was invisible until a smoke test actually
  built the agent under each of the 10 injections in a loop — unit tests exercised `naive_cases`,
  `case_result_rows`, and `naive_regressions` in isolation but never called `_build_adapter`. Fixed
  by merging into one dict (`{"temperature": cfg.temperature, **build_kwargs}`) so the injection's
  value wins instead of erroring; added `test_build_adapter_works_for_every_injection` and a
  dedicated regression test for the exact collision so this can't silently return.

**Still open — live execution deferred, by explicit user choice, not run silently:**

- Estimated cost of the full study (10 injections x 2 modes x 15 cases, plus 2 shared baselines):
  roughly 3,300 Gemini calls total (full-mode runs use each case's own trial count — 3-4 on
  average, with judge calls on 8 of 15 cases; naive-mode runs are 15 cases x 1 trial each, no
  judge). At the 15-requests/minute free-tier cap actually observed while seeding the Phase 8
  baseline, that's 3+ hours of pure throughput assuming zero rate-limit backoff — roughly 10x the
  call volume of the single `--trials 1` run that already hit that cap partway through.
- Options discussed: shrink the case/injection count for the study, switch to a paid Gemini tier
  for this one run, or run the full scope anyway spread across multiple quota windows. Decided:
  defer live execution: user will specify a reduced scope (fewer cases and/or fewer injections)
  before triggering a real run. `study/RESULTS.md` currently documents this pending state rather
  than fabricated numbers.
- README's study table, the "EvalGate ~9/10, naive ~4-6/10" comparison, and the v0.1.0 tag all
  depend on `study/RESULTS.md` having real numbers — none of those are done yet either, for the
  same reason.

---

## 2026-07-23 — Phase 9 addendum: first live study attempt, stopped at 7/10 (daily quota)

**What happened:** ran `study/run_study.py` for real against a 5-case subset (`nvda_cik_lookup`,
`nvda_latest_revenue`, `tsla_latest_revenue`, `msft_revenue_contains`, `format_bullet_list` —
chosen to cover every scorer type while avoiding `nvda_aapl_comparison`'s known pre-existing
defect and `flaky_ticker_only`'s history of quota-induced failures). Launched detached (`nohup` +
`disown`, not the harness's own background-task tracking) since the Bash tool's timeout caps at
10 minutes and this run was expected to take much longer — a harness-tracked background run got
killed by that cap on the first attempt before this was caught.

**Bugs encountered:**

- **First launch failed immediately: `ModuleNotFoundError: No module named 'examples'`.** Ran
  `python study/run_study.py` directly — running a `.py` file as a script only adds *that file's
  own directory* to `sys.path`, not the repository root, so `examples.sec_agent` (a namespace
  package living at the repo root, not installed) couldn't be found. `python -m study.run_study`
  fixes it: `-m` adds the current working directory to `sys.path`, matching how pytest already
  resolves the same import in the test suite. Worth remembering: any future study/study-adjacent
  script needs `-m` invocation from the repo root, not direct execution.

- **Ran the harness's own `run_in_background` once before realizing its 10-minute cap would kill
  a multi-hour run.** Switched to a fully detached process (`nohup ... &; disown`) plus a separate
  `Monitor` tail-and-filter on the log file, which survives independently of any single tool call's
  timeout.

- **First two monitor filter attempts leaked raw rate-limit tracebacks into near-continuous
  notifications.** The runner's per-trial retry-then-fail logic means a quota-exhausted stretch
  produces one multi-line Python traceback per failed call — with the study hitting the daily cap,
  that was hundreds of near-identical tracebacks. Fixed by having the log-tailing `awk` filter
  swallow every traceback/`RESOURCE_EXHAUSTED`/`ClientError` line into a running counter and only
  surface it as a one-line tally attached to the next real progress marker (`Injection #`,
  `caught:`, `Running ... baseline`). Real signal (progress + pass/fail) stayed visible; retry
  noise stopped flooding the conversation.

**The actual finding — Gemini's free tier has TWO separate quotas, and we'd only budgeted for
one.** The 429 error bodies distinguish `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`
(quotaValue 15, seen throughout Phase 8) from `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
(quotaValue **500**, first seen here). By the time injection #8 started, the log showed 1,164
hits against the *daily* 500-request cap — meaning the account had made several multiples of 500
requests already today (this study plus whatever else ran earlier), and every 429 from that point
on was unrecoverable within the same day. The runner's one-retry-with-60s-backoff logic is
correctly designed for the *per-minute* cap (waiting out a short window) but can do nothing about
a day-level cap — it will just keep retrying and failing until the day rolls over. This wasn't a
bug in the code; it's a cost-planning gap: the ~3,300-call estimate in `NOTES.md`'s prior Phase 9
entry accounted for total call volume but not for the fact that the free tier's real ceiling is
whichever of the two quotas is tighter on a given day, and 500/day is far tighter than
15/minute × however many minutes a multi-hour run spans.

**Decision:** stopped the process at injection #8 rather than let it burn the rest of the day's
(already-exhausted) quota on trials that could only fail. Kept the 7/10 completed injections'
results in `study/RESULTS.md` rather than discarding them — real signal, worth having even
incomplete. Full analysis and the remaining 3 injections (#8 skip_analysis_for_comparisons, #9
corrupt_fixture_revenue, #10 one_sentence_report) are deferred until quota resets.

**Early, provisional observation (do not treat as final — see caveats in RESULTS.md):** full mode
and naive mode agreed on all 7 completed injections (5 caught, 2 missed) — no case yet where
EvalGate's N-trial statistics or CI logic caught something a naive single trial missed. The 2
misses (#1 no-citation, #2 prefer-annual-data) are both prompt-quality regressions that none of
the 5 subset cases' scorers happened to check for, which may be an artifact of this specific
5-case subset rather than a real finding about EvalGate vs. naive mode — worth re-checking once
the study runs against the full 15-case set.

---

## 2026-09-23 — Phase 9 completion: full 15-case study, real numbers, two infra fixes

**Context:** four days before an interview where this project needed to be demoable, so the
priority shifted from "cheapest possible completion" to "correct, complete result, fast." That
reprioritization drove every decision below — several alternatives were explored and abandoned
specifically because they cost more time than they saved.

**Key decisions:**

- **Paid Gemini tier, not a free alternative provider.** Explored Groq (free tier: Llama 3.x
  models are retired; the only tool-calling-capable models left, `openai/gpt-oss-120b`/`20b`, cap
  at 1,000 requests/day *and* a tighter-than-expected 8,000 tokens/minute — the latter alone would
  need ~28 hours of continuous throughput for this study's ~13M-token volume) and Ollama running
  locally (`llama3.1:8b` on the dev machine's 16GB M4: zero external rate cap, but it concretely
  broke the multi-agent hand-off protocol on the very first smoke-test trial and hallucinated a
  revenue figure off by roughly 10x). Both are recorded here because they're legitimate rejected
  alternatives, not because either shipped. Real per-token pricing pulled live from
  `ai.google.dev/gemini-api/docs/pricing` put the full 15-case study at **~$3.55** on
  `gemini-3.1-flash-lite` — cheap enough that "pay for it" beat "debug a smaller model's
  reliability under time pressure." (`evalgate.toml`'s configured cost rates, $0.075/$0.30 per 1M,
  are stale against this real pricing — a leftover from whenever they were last set; worth fixing
  separately, not urgent enough to block this run.)

- **`max_concurrent_trials` reverted from 6 back to 1 after one failed attempt, not debugged
  further.** Paid tier easily clears the RPM/RPD math that originally forced serialization, so 6
  was tried once to speed things up. It triggered a burst of ADK `ValueError: Tool 'X' not found`
  errors that the serialized rerun did *not* reproduce at the same rate in its first minutes (see
  below — a lower rate of the same error did eventually show up even at concurrency 1, so the
  causal link to concurrency specifically was never proven, only suspected). Given the interview
  deadline, "revert to the config already known to work" was the correct call over spending time
  proving or disproving an ADK concurrency-safety hypothesis that doesn't change what to *do*
  either way.

- **Runner retry logic broadened from rate-limit-only to rate-limit-or-transient-server-error.**
  `runner.py`'s one-retry-then-fail path only recognized `RESOURCE_EXHAUSTED`/`429` (Phase 4
  addendum's free-tier design). Paid tier doesn't hit those, but a real Gemini 503
  ("the service is currently unavailable") surfaced mid-run and was *not* retried — it survived
  even the `google-genai` SDK's own internal `tenacity` retries before reaching evalgate's runner.
  Fixed by adding `"503"`, `"UNAVAILABLE"`, and `"ServerError"` to the retry-worthy marker list
  alongside the existing rate-limit markers, and mirroring the same broadened list in
  `study/run_study.py`'s `_check_for_quota_failures` — a round tainted by an unretried 503 would
  otherwise have been silently checkpointed as if it were real agent behavior. Added
  `tests/test_runner.py` (previously zero unit coverage on this retry path, 429 case included) and
  a new `_check_for_quota_failures` case for the 503 marker.

**Bugs encountered:**

- **ADK `ValueError: Tool 'retrieval_agent' not found` (and, once, `'retrieval_agent:get_company_facts'`,
  `'default_agent:report_agent'`).** Traced into `google/adk/flows/llm_flows/functions.py`'s
  `_get_tool()` — its own error message names the likely cause: "LLM hallucinated the function
  name." The coordinator's only real tool is `transfer_to_agent`; `gemini-3.1-flash-lite`
  occasionally calls a sub-agent's *name* directly instead, as if it were a callable tool. Not a
  bug in this codebase or in ADK — genuine, if intermittent, small-model tool-calling unreliability,
  and exactly the kind of thing the multi-trial statistical framework exists to quantify honestly
  rather than hide. One specific consequence worth remembering: `flaky_ticker_only`
  (`examples/sec_agent/cases/flaky_ticker_only.yaml`) was designed to test *formatting* flakiness
  (a CIK→ticker question with a strict one-word regex), but its actual observed failure mode in
  this run was this tool-hallucination bug instead — the agent has no CIK→ticker tool at all, so
  the model's most common mistake is passing the CIK into `lookup_cik(ticker=...)`, which correctly
  raises `ValueError: Ticker '...' not found`, but ADK propagates that as an uncaught exception
  instead of handing it back to the model to recover from. The case still produces valid signal,
  just via a different mechanism than the one it was written to exercise.

- **Two prior live-run attempts had to be discarded before this one, both caught before any bad
  data was checkpointed.** (1) The `max_concurrent_trials=6` attempt above. (2) A run that hit an
  unretried 503 mid-injection — confirmed via the checkpoint file's contents that the baseline data
  already saved was clean (computed before the 503 occurred), so nothing false was ever persisted;
  the in-progress round that hit the 503 was correctly never checkpointed, matching the Phase 9
  "quota safety" design intent even though 503 wasn't yet in scope for it.

**Final result — full 15-case set, single clean run, ~$3.55, ~50 minutes wall-clock at
`max_concurrent_trials=1`:**

EvalGate (full mode) caught **7/10**. Naive mode (1 trial, exact-match-only) caught **6/10**.
The one divergence — **Injection #1 (report_agent's citation requirement deleted) was caught by
full mode and missed by naive mode** — is the concrete demonstration the whole study exists to
produce: naive single-trial exact-match testing missed a real prompt-quality regression that
EvalGate's judge scorer plus multi-trial statistics caught. Both modes missed #5 (temperature
raised to 1.0 — a noise/consistency regression with no scorer built to catch it), #8 (coordinator
skips the analysis agent for comparisons), and #10 (forced one-sentence answers) — none of the 15
cases' scorers were written to check for reasoning-consistency or answer-completeness, which is an
honest scorer-coverage gap worth naming if asked, not a framework failure. Full table in
`study/RESULTS.md`. This replaces the earlier partial 7/10-of-10-injections-on-a-5-case-subset
result from the prior addendum — that one is now superseded, not merged.
