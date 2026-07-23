# Regression-Injection Study Results

**Status: partial — 7/10 injections complete.** Run against a reduced 5-case subset
(`nvda_cik_lookup`, `nvda_latest_revenue`, `tsla_latest_revenue`, `msft_revenue_contains`,
`format_bullet_list` — chosen to cover numeric/trajectory/judge/contains/regex scoring and to
avoid `nvda_aapl_comparison`, which has a pre-existing real agent defect unrelated to this study).
Stopped partway through injection #8 after exhausting the day's Gemini free-tier daily quota (500
requests/day on `gemini-3.1-flash-lite`) — see `NOTES.md` for the full account. Resume with:

```bash
python -m study.run_study --cases <same 5-case subset dir> --output study/RESULTS.md
```

once quota resets, picking up from injection #8.

EvalGate (full mode) caught **5/7** so far. Naive mode (1 trial, exact-match-only) caught **5/7**
— identical to full mode on every injection completed so far, which is itself a finding (see
below).

| # | Injection | EvalGate (full) | Naive (1 trial) |
|---|-----------|:---:|:---:|
| 1 | Delete the citation requirement from report_agent | missed | missed |
| 2 | retrieval_agent prefers 10-K annual totals when asked for quarterly data | missed | missed |
| 3 | Context trimming returns the second-most-recent quarter as most recent | caught | caught |
| 4 | get_recent_filings removed from retrieval_agent's tools | caught | caught |
| 5 | Agent temperature raised to 1.0 (noise increase) | caught | caught |
| 6 | Tool output truncated to 500 chars before returning to the model | caught | caught |
| 7 | calculate() silently rounds to the nearest billion | caught | caught |
| 8 | Coordinator skips the analysis agent for comparison questions | not run | not run |
| 9 | One fixture's revenue value corrupted by +10% | not run | not run |
| 10 | report_agent forced to answer in exactly one sentence | not run | not run |

## Early observations (7/10, not yet a complete result)

- **Both modes agree on every injection so far — no case yet where full mode's statistical/CI
  machinery caught something naive mode's single trial missed, or vice versa.** This isn't
  necessarily surprising at n=7 on a 5-case subset: the injections that *should* separate the two
  modes hardest are the noise-flavored ones (#5, temperature) and the ones that only show up
  probabilistically across trials rather than deterministically breaking output — #5 was in fact
  caught by both, which is worth a closer look once the full study reruns: was it caught by real
  signal, or partly by quota-exhaustion noise inflating apparent failure rates in a way that
  happened to look like a regression in both modes? Should not be trusted until re-verified on a
  clean quota day.
- **#1 and #2 (prompt-only regressions) were missed by both modes on this subset.** Both are
  instruction changes that degrade a *quality* dimension (citation completeness, data recency)
  that none of the 5 chosen cases' deterministic scorers (contains/regex/numeric) directly check,
  and none of the 5 cases happened to have a judge rubric that caught it either. This may be a
  property of the specific 5-case subset rather than of EvalGate's design — the missing 3 cases in
  the full 15 include ones with a judge rubric specifically about citation quality
  (`msft_revenue_contains` has one, and still missed it, worth investigating further once judge
  outputs from this run can be inspected in the DB).
- No conclusions about EvalGate vs. naive mode should be drawn from this partial run — 7 data
  points on 5 cases is not the full picture. Full analysis to follow once injections 8-10 complete
  and (ideally) the study is re-run on the full 15-case set.
