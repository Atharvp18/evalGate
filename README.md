# EvalGate

An evaluation and regression-testing framework for LLM agent systems — like pytest, but for agent behavior.

EvalGate runs test cases N times (because LLM output is non-deterministic), scores results statistically, and blocks GitHub PRs that silently regress agent quality.

The framework is exercised against a **SEC-filings research assistant** built on Google ADK that answers questions about public company filings using the free SEC EDGAR API.

---

## Setup

```bash
# Clone and enter the repo
git clone git@github.com:Atharvp18/evalGate.git
cd evalGate

# Install dependencies
/opt/homebrew/bin/uv sync

# Activate the virtual environment
source .venv/bin/activate

# Add your Google API key (get one from aistudio.google.com)
echo "GOOGLE_API_KEY=your_key_here" > .env

# Update your email in evalgate.toml (required by SEC EDGAR)
# Set: user_agent = "EvalGate research project <your@email.com>"
```

---

## Commands

**Run the SEC agent interactively (live mode):**
```bash
python -m examples.sec_agent.chat
```
Ask it anything about public companies:
- *"What was Nvidia's revenue in its most recent quarter?"*
- *"Compare Apple and Microsoft revenue growth over the last year"*
- *"What was Tesla's net income in its latest 10-Q?"*

**Run all tests (no network, no API calls needed):**
```bash
python -m pytest -v
```

**Re-record EDGAR fixtures** (only needed if EDGAR data is stale):
```bash
python examples/sec_agent/record_fixtures.py
```

**Check code quality:**
```bash
ruff check src/ tests/ examples/
```

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold — pyproject, config, CLI stub | ✅ Done |
| 1 | EDGAR client with live/replay/record modes + fixtures | ✅ Done |
| 2 | SEC agent (ADK coordinator + 3 sub-agents) + ADKAdapter | ✅ Done |
| 3 | Schema, YAML loader, async N-trial runner | 🔜 Next |
| 4 | Scorers (contains, regex, numeric, trajectory, judge) | — |
| 5 | Wilson CI stats, SQLite store, `evalgate report` | — |
| 6 | pytest plugin, baseline save, regression gate | — |
| 7 | Judge calibration (Cohen's kappa) | — |
| 8 | GitHub Actions CI gate, trace mining, dashboard | — |
| 9 | Regression-injection study + release polish | — |

---

## Architecture

```
evalgate/
├── src/evalgate/        # The framework (library + CLI)
│   ├── adapter.py       # AgentAdapter protocol (framework-agnostic)
│   ├── adapters/adk.py  # ADK implementation
│   ├── config.py        # evalgate.toml loader
│   └── cli.py           # typer CLI
└── examples/sec_agent/  # Subject agent (what gets tested)
    ├── agent.py          # ADK coordinator + 3 sub-agents
    ├── tools/edgar.py    # EDGAR client + tool functions
    ├── fixtures/         # Frozen EDGAR responses (committed)
    └── chat.py           # Interactive REPL
```
