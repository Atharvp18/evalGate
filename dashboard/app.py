"""Streamlit dashboard — three views: pass-rate over runs, latest-run table, cost & latency.

Run with: streamlit run dashboard/app.py -- --db evalgate.db
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evalgate.store import connect


def _parse_db_path() -> str:
    """Read --db from the args Streamlit passes through after `--`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="evalgate.db")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.db


@st.cache_data(ttl=30)
def _load_tables(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three tables the dashboard needs as plain DataFrames.

    Cached for 30s so repeated widget interactions (multiselect changes, tab
    switches) don't re-open the DB on every rerun — refresh is one page reload
    away, which is all a local eval-review tool needs.
    """
    conn = connect(db_path)
    try:
        runs = pd.read_sql_query("SELECT * FROM runs ORDER BY id", conn)
        cases = pd.read_sql_query(
            "SELECT cr.*, r.started_at FROM case_results cr "
            "JOIN runs r ON cr.run_id = r.id ORDER BY cr.run_id, cr.case_id",
            conn,
        )
        trials = pd.read_sql_query(
            "SELECT t.*, cr.run_id, cr.case_id FROM trials t "
            "JOIN case_results cr ON t.case_result_id = cr.id",
            conn,
        )
    finally:
        conn.close()
    return runs, cases, trials


def _render_pass_rate_tab(cases: pd.DataFrame) -> None:
    st.subheader("Pass rate over runs, per case (with 95% Wilson CI band)")
    case_ids = sorted(cases["case_id"].unique())
    selected = st.multiselect("Cases", case_ids, default=case_ids[: min(5, len(case_ids))])

    fig = go.Figure()
    for case_id in selected:
        sub = cases[cases["case_id"] == case_id].sort_values("run_id")
        fig.add_trace(
            go.Scatter(x=sub["run_id"], y=sub["pass_rate"], mode="lines+markers", name=case_id)
        )
        # CI band: draw the high edge forward and the low edge backward as one
        # closed polygon, so `fill="toself"` shades the region between them.
        fig.add_trace(
            go.Scatter(
                x=pd.concat([sub["run_id"], sub["run_id"][::-1]]),
                y=pd.concat([sub["ci_high"], sub["ci_low"][::-1]]),
                fill="toself",
                fillcolor="rgba(128,128,128,0.15)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name=f"{case_id} CI",
            )
        )
    fig.update_layout(xaxis_title="Run ID", yaxis_title="Pass rate", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def _render_latest_run_tab(cases: pd.DataFrame) -> None:
    st.subheader("Latest run")
    latest_run_id = int(cases["run_id"].max())
    latest = cases[cases["run_id"] == latest_run_id].copy()
    latest["ci_95"] = latest.apply(lambda r: f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]", axis=1)
    latest["flaky"] = latest["flaky"].map({1: "yes", 0: ""})
    latest["threshold"] = latest["passed_threshold"].map({1: "ok", 0: "FAIL"})
    st.dataframe(
        latest[["case_id", "trials", "passes", "pass_rate", "ci_95", "flaky", "threshold"]],
        use_container_width=True,
        hide_index=True,
    )


def _render_cost_latency_tab(trials: pd.DataFrame) -> None:
    st.subheader("Cost & latency over runs")
    agg = (
        trials.groupby("run_id")
        .agg(
            total_cost_usd=("cost_usd", "sum"),
            p50_latency_ms=("latency_ms", "median"),
            p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
        )
        .reset_index()
    )
    st.plotly_chart(
        px.line(
            agg, x="run_id", y="total_cost_usd", markers=True, title="Total cost per run (USD)"
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(
            agg,
            x="run_id",
            y=["p50_latency_ms", "p95_latency_ms"],
            markers=True,
            title="Latency per run (ms)",
        ),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="EvalGate Dashboard", layout="wide")
    st.title("EvalGate Dashboard")

    db_path = _parse_db_path()
    runs, cases, trials = _load_tables(db_path)

    if runs.empty:
        st.warning(f"No runs found in {db_path}. Run `evalgate run` first.")
        return

    st.caption(f"Reading `{db_path}` — {len(runs)} run(s), {cases['case_id'].nunique()} case(s)")

    tab1, tab2, tab3 = st.tabs(["Pass rate over runs", "Latest run", "Cost & latency"])
    with tab1:
        _render_pass_rate_tab(cases)
    with tab2:
        _render_latest_run_tab(cases)
    with tab3:
        _render_cost_latency_tab(trials)


if __name__ == "__main__":
    main()
