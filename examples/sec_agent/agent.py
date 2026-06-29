"""SEC-filings research assistant: ADK coordinator + 3 sub-agents.

Architecture:
    coordinator (LlmAgent)
        ├── retrieval_agent  — owns the three EDGAR tools
        ├── analysis_agent   — does the arithmetic with `calculate`
        └── report_agent     — formats the final cited answer

Prompts are kept short, in Python constants, and version-controlled. Prompt
changes are exactly what the regression gate will police later, so every
edit lives in git.

Build with `build_agent()`. The function returns the coordinator; the EDGAR
client must already be configured via tools.edgar.configure_client() before
the agent is actually run.
"""

from __future__ import annotations

import ast
import operator as op

from examples.sec_agent.tools.edgar import (
    get_company_facts,
    get_recent_filings,
    lookup_cik,
)
from google.adk.agents import LlmAgent

# Default model. Picked because the AI Studio free tier currently includes
# gemini-2.5-flash. Override per-agent for experimentation.
DEFAULT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Safe calculator tool for the analysis agent.
# ---------------------------------------------------------------------------

_SAFE_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv,
}
_SAFE_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed expression node: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Evaluate a pure arithmetic expression and return the numeric result.

    Supports +, -, *, /, %, **, // and parentheses. No variables, no function
    calls, no attribute access. Use this for growth rates, ratios, and
    percentage changes computed from values returned by retrieval tools.

    Args:
        expression: Arithmetic expression, e.g. "(81615 - 60922) / 60922 * 100".

    Returns:
        dict with keys:
          - expression (str): the input.
          - result (float): the evaluated value, or None on error.
          - error (str | None): non-null only on a parse/evaluation failure.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree)
        return {"expression": expression, "result": value, "error": None}
    except Exception as exc:
        return {"expression": expression, "result": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Sub-agent instructions
# ---------------------------------------------------------------------------

RETRIEVAL_INSTRUCTION = """\
You are the retrieval agent. Your job is to find SEC EDGAR data the team needs.

Workflow:
1. If the user mentions a company by ticker or name, call lookup_cik(ticker)
   to obtain the CIK number. Always do this first when you do not have a CIK.
2. Use get_company_facts(cik) to fetch trimmed quarterly/annual financials
   (revenue, net income, EPS, assets, shares).
3. Use get_recent_filings(cik, form_type) only if the user asks about specific
   filings, accession numbers, or filing dates.

Rules:
- Fetch ONLY the companies and concepts the question requires. Do not pre-fetch
  comparison companies "in case" — that wastes context.
- Return the raw structured data you retrieved. Do NOT compute growth rates or
  comparisons yourself; that is the analysis agent's job.
- If a tool errors, report the error verbatim so the coordinator can decide
  whether to try a different ticker or stop.
"""

ANALYSIS_INSTRUCTION = """\
You are the analysis agent. You receive structured financial data and a
question; you produce computed answers using the `calculate` tool.

Rules:
- For every numeric answer, use `calculate` with an explicit arithmetic
  expression. Never compute in your head.
- When comparing two companies, compute growth rates or ratios in the units
  the user expects (percentage points, USD billions, etc.).
- Cite the exact period_end dates and forms (10-Q / 10-K) of the source
  values in your reasoning so the report agent can attribute them.
- If the data needed to answer is not present, say so plainly — do not guess.
"""

REPORT_INSTRUCTION = """\
You are the report agent. You produce the final user-facing answer.

Rules:
- Use the company name and the source filing (e.g. "10-Q, period ending
  2026-04-26") for every figure. Always cite.
- NEVER invent numbers, dates, tickers, or CIKs. If a value was not retrieved,
  say so explicitly rather than approximating.
- Format large numbers with units ("$81.6 billion", not "81615000000").
- Keep the answer focused on what the user asked. No filler.
"""

COORDINATOR_INSTRUCTION = """\
You coordinate a team that answers questions about US public companies using
SEC EDGAR filings.

You have three sub-agents you can transfer to:
- retrieval_agent: fetches CIKs, company facts, and filings metadata.
- analysis_agent: does arithmetic on retrieved values.
- report_agent: writes the final cited answer.

Standard flow for a financial question:
1. Transfer to retrieval_agent to fetch the underlying data.
2. If math is needed (growth rates, ratios, comparisons), transfer to
   analysis_agent.
3. Always transfer to report_agent for the final answer, with the source
   data and computed values available in context.

If the question can be answered without analysis (e.g. "what was X's CIK?"),
skip step 2.
"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_agent(model: str = DEFAULT_MODEL) -> LlmAgent:
    """Construct the coordinator with its three sub-agents wired in.

    The EDGAR client must be configured (via
    `examples.sec_agent.tools.edgar.configure_client`) before invoking the
    returned agent — the tools call into a module-level singleton.
    """
    retrieval_agent = LlmAgent(
        name="retrieval_agent",
        model=model,
        description="Fetches SEC EDGAR data: CIKs, company facts, recent filings.",
        instruction=RETRIEVAL_INSTRUCTION,
        tools=[lookup_cik, get_company_facts, get_recent_filings],
    )

    analysis_agent = LlmAgent(
        name="analysis_agent",
        model=model,
        description="Performs arithmetic on retrieved financial figures.",
        instruction=ANALYSIS_INSTRUCTION,
        tools=[calculate],
    )

    report_agent = LlmAgent(
        name="report_agent",
        model=model,
        description="Writes the final cited answer for the user.",
        instruction=REPORT_INSTRUCTION,
        tools=[],
    )

    return LlmAgent(
        name="sec_coordinator",
        model=model,
        description="Routes SEC questions across retrieval, analysis, and report sub-agents.",
        instruction=COORDINATOR_INSTRUCTION,
        sub_agents=[retrieval_agent, analysis_agent, report_agent],
    )
