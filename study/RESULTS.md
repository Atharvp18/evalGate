# Regression-Injection Study Results

**Status: pending.** The study code (`study/injections.py`, `study/run_study.py`) is built and
unit-tested — every injection's monkeypatch mechanics are verified in isolation, and
`study/run_study.py --cases <subset> --output study/RESULTS.md` is ready to run. It has not yet
been executed against the live agent.

**Why it's pending:** a full run (10 injections x 2 modes x 15 cases, plus 2 baselines) is an
estimated ~3,300 Gemini calls. At the free-tier cap actually observed while seeding the Phase 8
baseline (15 requests/minute), that's 3+ hours of pure throughput with zero rate-limit waiting —
not something to run silently. See `NOTES.md` (Phase 9 entry) for the full cost breakdown and the
options discussed.

**Next step:** run it against a reduced case subset (fewer than all 15 cases and/or fewer than
all 10 injections) once the scope is picked, e.g.:

```bash
python study/run_study.py --cases examples/sec_agent/cases --output study/RESULTS.md
```

This file will be overwritten with the real catch-rate table once that run completes.
