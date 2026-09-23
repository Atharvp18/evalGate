# Regression-Injection Study Results

EvalGate (full mode) caught **7/10**. Naive mode (1 trial, exact-match-only) caught **6/10**.

| # | Injection | EvalGate (full) | Naive (1 trial) |
|---|-----------|:---:|:---:|
| 1 | Delete the citation requirement from report_agent | caught | missed |
| 2 | retrieval_agent prefers 10-K annual totals when asked for quarterly data | caught | caught |
| 3 | Context trimming returns the second-most-recent quarter as most recent | caught | caught |
| 4 | get_recent_filings removed from retrieval_agent's tools | caught | caught |
| 5 | Agent temperature raised to 1.0 (noise increase) | missed | missed |
| 6 | Tool output truncated to 500 chars before returning to the model | caught | caught |
| 7 | calculate() silently rounds to the nearest billion | caught | caught |
| 8 | Coordinator skips the analysis agent for comparison questions | missed | missed |
| 9 | One fixture's revenue value corrupted by +10% | caught | caught |
| 10 | report_agent forced to answer in exactly one sentence | missed | missed |
