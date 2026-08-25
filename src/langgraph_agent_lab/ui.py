"""Streamlit dashboard for live LangGraph lab demonstrations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import sleep
from typing import cast
from urllib.parse import urlparse
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import (
    MetricsReport,
    ScenarioMetric,
    metric_from_state,
    summarize_metrics,
    write_metrics,
)
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.report import render_report, write_report
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import AgentState, Route, Scenario, initial_state

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
SAMPLE_SCENARIOS = ROOT / "data" / "sample" / "scenarios.jsonl"
UI_CHECKPOINTS = ROOT / "outputs" / "ui_checkpoints.db"
UI_METRICS = ROOT / "outputs" / "ui_metrics.json"
UI_REPORT = ROOT / "reports" / "ui_lab_report.md"

PRESETS: dict[str, dict[str, object]] = {
    "Simple question": {
        "query": "How do I reset my password?",
        "route": Route.SIMPLE,
        "max_attempts": 3,
    },
    "Tool lookup": {
        "query": "Please lookup order status for order 12345",
        "route": Route.TOOL,
        "max_attempts": 3,
    },
    "Missing information": {
        "query": "Can you fix it?",
        "route": Route.MISSING_INFO,
        "max_attempts": 3,
    },
    "Risky refund + HITL": {
        "query": "Refund this customer and send confirmation email",
        "route": Route.RISKY,
        "max_attempts": 3,
    },
    "Transient error + recovery": {
        "query": "Timeout failure while processing request",
        "route": Route.ERROR,
        "max_attempts": 3,
    },
    "Dead letter": {
        "query": "System failure cannot recover after multiple attempts",
        "route": Route.ERROR,
        "max_attempts": 1,
    },
    "Custom": {"query": "", "route": Route.SIMPLE, "max_attempts": 3},
}

GRAPH_DOT = """
digraph LangGraph {
  rankdir=LR;
  graph [bgcolor="transparent", pad="0.3", nodesep="0.35", ranksep="0.6"];
  node [shape=box, style="rounded,filled", fillcolor="#F4F7FF", color="#6C7AE0",
        fontname="Arial", fontsize=10];
  edge [color="#667085", fontname="Arial", fontsize=9];
  START [shape=circle, fillcolor="#D1FADF", color="#12B76A"];
  END [shape=doublecircle, fillcolor="#D1FADF", color="#12B76A"];
  classify [fillcolor="#E0EAFF", color="#6172F3"];
  approval [fillcolor="#FEF0C7", color="#F79009"];
  retry [fillcolor="#FEE4E2", color="#F04438"];
  dead_letter [fillcolor="#FEE4E2", color="#D92D20"];
  START -> intake -> classify;
  classify -> answer [label="simple"];
  classify -> tool [label="tool"];
  classify -> clarify [label="missing_info"];
  classify -> risky_action [label="risky"];
  classify -> retry [label="error"];
  risky_action -> approval;
  approval -> tool [label="approved"];
  approval -> clarify [label="rejected"];
  tool -> evaluate;
  evaluate -> answer [label="success"];
  evaluate -> retry [label="needs_retry"];
  retry -> tool [label="within budget"];
  retry -> dead_letter [label="exhausted"];
  answer -> finalize;
  clarify -> finalize;
  dead_letter -> finalize;
  finalize -> END;
}
"""

GRAPH_NODES = (
    "START",
    "intake",
    "classify",
    "answer",
    "tool",
    "evaluate",
    "clarify",
    "risky_action",
    "approval",
    "retry",
    "dead_letter",
    "finalize",
    "END",
)

GRAPH_EDGES = (
    ("START", "intake", ""),
    ("intake", "classify", ""),
    ("classify", "answer", "simple"),
    ("classify", "tool", "tool"),
    ("classify", "clarify", "missing_info"),
    ("classify", "risky_action", "risky"),
    ("classify", "retry", "error"),
    ("risky_action", "approval", ""),
    ("approval", "tool", "approved"),
    ("approval", "clarify", "rejected"),
    ("tool", "evaluate", ""),
    ("evaluate", "answer", "success"),
    ("evaluate", "retry", "needs_retry"),
    ("retry", "tool", "within budget"),
    ("retry", "dead_letter", "exhausted"),
    ("answer", "finalize", ""),
    ("clarify", "finalize", ""),
    ("dead_letter", "finalize", ""),
    ("finalize", "END", ""),
)


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px;}
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #ffffff 0%, #f7f9ff 100%);
            border: 1px solid #e4e7ec; border-radius: 14px; padding: 12px 16px;
            box-shadow: 0 3px 12px rgba(16,24,40,.05);
        }
        .hero {
            padding: 22px 26px; border-radius: 18px; margin-bottom: 18px;
            background: linear-gradient(120deg, #182230 0%, #3448c5 65%, #6172f3 100%);
            color: white; box-shadow: 0 12px 30px rgba(52,72,197,.22);
        }
        .hero h1 {margin: 0 0 5px 0; font-size: 2rem; color: white;}
        .hero p {margin: 0; opacity: .86;}
        .pill {display:inline-block; padding:4px 10px; border-radius:999px;
               background:#eef4ff; color:#3538cd; font-weight:600; margin-right:6px;}
        .trace-title {font-weight:700; color:#344054; margin-top:8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _cached_graph(checkpointer_kind: str) -> CompiledStateGraph:
    database_url = str(UI_CHECKPOINTS) if checkpointer_kind == "sqlite" else None
    return build_graph(build_checkpointer(checkpointer_kind, database_url))


def _configure_hitl(enabled: bool) -> str | None:
    previous = os.getenv("LANGGRAPH_INTERRUPT")
    os.environ["LANGGRAPH_INTERRUPT"] = "true" if enabled else "false"
    return previous


def _restore_hitl(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("LANGGRAPH_INTERRUPT", None)
    else:
        os.environ["LANGGRAPH_INTERRUPT"] = previous


def _invoke(
    graph: CompiledStateGraph,
    graph_input: AgentState | Command,
    config: RunnableConfig,
    *,
    real_hitl: bool,
) -> AgentState:
    previous = _configure_hitl(real_hitl)
    try:
        return cast(AgentState, graph.invoke(graph_input, config=config))
    finally:
        _restore_hitl(previous)


def _event_rows(state: AgentState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, event in enumerate(state.get("events", []), start=1):
        metadata = event.get("metadata", {}) or {}
        rows.append(
            {
                "#": index,
                "node": event.get("node", "unknown"),
                "event": event.get("event_type", ""),
                "message": event.get("message", ""),
                "latency_ms": event.get("latency_ms", 0),
                "llm_calls": metadata.get("llm_calls", 0),
                "mode": metadata.get("evaluation_mode", ""),
            }
        )
    return rows


def _history_rows(graph: CompiledStateGraph, config: RunnableConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, snapshot in enumerate(graph.get_state_history(config), start=1):
        values = cast(dict[str, object], snapshot.values)
        rows.append(
            {
                "checkpoint": index,
                "created_at": snapshot.created_at or "",
                "next": ", ".join(snapshot.next) or "END",
                "route": values.get("route", ""),
                "attempt": values.get("attempt", 0),
                "events": len(cast(list[object], values.get("events", []))),
            }
        )
    return rows


def _has_interrupt(state: AgentState) -> bool:
    return bool(cast(dict[str, object], state).get("__interrupt__"))


def _execution_path(state: AgentState) -> list[str]:
    """Build the ordered node path represented by audit events and interrupts."""
    path = ["START"]
    path.extend(str(event.get("node", "unknown")) for event in state.get("events", []))
    if _has_interrupt(state) and (not path or path[-1] != "approval"):
        path.append("approval")
    elif path and path[-1] == "finalize":
        path.append("END")
    return path


def _dynamic_graph_dot(state: AgentState, upto_step: int | None = None) -> str:
    """Render the graph with executed nodes and edges highlighted."""
    full_path = _execution_path(state)
    final_step = len(full_path) - 1 if upto_step is None else min(upto_step, len(full_path) - 1)
    visible_path = full_path[: final_step + 1]
    visited = set(visible_path)
    current = visible_path[-1]
    traversed_edges = set(zip(visible_path, visible_path[1:], strict=False))

    lines = [
        "digraph LangGraph {",
        '  rankdir=LR; graph [bgcolor="transparent", pad="0.35", nodesep="0.35", ranksep="0.65"];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", '
        'fontsize=10, margin="0.12,0.08"];',
        '  edge [fontname="Arial", fontsize=9, arrowsize=0.75];',
    ]
    for node in GRAPH_NODES:
        shape = "doublecircle" if node == "END" else "circle" if node == "START" else "box"
        if node == current:
            fill, color, penwidth = "#FEC84B", "#DC6803", 3
        elif node in visited and node in {"retry", "dead_letter"}:
            fill, color, penwidth = "#FEE4E2", "#D92D20", 2
        elif node in visited and node == "approval":
            fill, color, penwidth = "#FEF0C7", "#F79009", 2
        elif node in visited:
            fill, color, penwidth = "#D1FADF", "#12B76A", 2
        else:
            fill, color, penwidth = "#F2F4F7", "#D0D5DD", 1
        label = node.replace("_", "\\n")
        lines.append(
            f'  {node} [label="{label}", shape={shape}, fillcolor="{fill}", '
            f'color="{color}", penwidth={penwidth}];'
        )

    for source, target, label in GRAPH_EDGES:
        if (source, target) in traversed_edges:
            color = "#D92D20" if target in {"retry", "dead_letter"} else "#3448C5"
            style = f'color="{color}", penwidth=3, fontcolor="{color}"'
        else:
            style = 'color="#D0D5DD", penwidth=1, fontcolor="#98A2B3"'
        label_attr = f', label="{label}"' if label else ""
        lines.append(f"  {source} -> {target} [{style}{label_attr}];")
    lines.append("}")
    return "\n".join(lines)


def _path_caption(path: list[str], upto_step: int | None = None) -> str:
    final_step = len(path) - 1 if upto_step is None else min(upto_step, len(path) - 1)
    return " → ".join(path[: final_step + 1])


def _render_result(state: AgentState, expected_route: str) -> None:
    metric = metric_from_state(
        cast(dict[str, object], state),
        expected_route=expected_route,
        approval_required=expected_route == Route.RISKY.value,
    )
    events = state.get("events", [])
    total_latency = sum(int(event.get("latency_ms", 0) or 0) for event in events)

    st.subheader("Live workflow map")
    st.graphviz_chart(_dynamic_graph_dot(state), width="stretch")
    st.caption(f"Executed path: {_path_caption(_execution_path(state))}")

    cols = st.columns(6)
    cols[0].metric("Actual route", state.get("route", "pending") or "pending")
    cols[1].metric("Expected", expected_route)
    cols[2].metric("Nodes", len(events))
    cols[3].metric("Retries", metric.retry_count)
    cols[4].metric("LLM calls", metric.llm_calls)
    cols[5].metric("LLM latency", f"{total_latency:,} ms")

    if _has_interrupt(state):
        st.warning("Workflow is paused at the approval node and waiting for a human decision.")
    elif metric.success:
        st.success("Workflow completed with the expected behavior.")
    else:
        st.error("Workflow completed, but its behavior differs from the expected route/outcome.")

    answer = state.get("final_answer") or state.get("pending_question")
    if answer:
        st.subheader("Final response")
        st.info(answer)

    if state.get("proposed_action"):
        st.subheader("Proposed risky action")
        st.warning(state["proposed_action"])

    st.subheader("Execution trace")
    st.dataframe(_event_rows(state), width="stretch", hide_index=True)

    detail_tabs = st.tabs(["Tool results", "Errors", "State JSON", "Messages"])
    with detail_tabs[0]:
        tool_results = state.get("tool_results", [])
        st.code("\n\n".join(tool_results) if tool_results else "No tool result", language="text")
    with detail_tabs[1]:
        errors = state.get("errors", [])
        st.code("\n".join(errors) if errors else "No errors", language="text")
    with detail_tabs[2]:
        st.json(cast(dict[str, object], state), expanded=False)
    with detail_tabs[3]:
        st.code("\n".join(state.get("messages", [])) or "No messages", language="text")


def _render_pending_approval() -> None:
    if "pending_graph" not in st.session_state:
        return
    st.subheader("Human approval console")
    st.caption("This resumes the same checkpointed thread; it does not restart the graph.")
    comment = st.text_input("Reviewer comment", value="Reviewed during live demo")
    approve_col, reject_col, _ = st.columns([1, 1, 3])
    approved = approve_col.button("Approve", type="primary", width="stretch")
    rejected = reject_col.button("Reject", width="stretch")
    if not (approved or rejected):
        return

    graph = cast(CompiledStateGraph, st.session_state["pending_graph"])
    config = cast(RunnableConfig, st.session_state["pending_config"])
    decision = {
        "approved": approved,
        "reviewer": "streamlit-reviewer",
        "comment": comment,
    }
    with st.spinner("Resuming checkpointed workflow..."):
        result = _invoke(graph, Command(resume=decision), config, real_hitl=True)
    st.session_state["last_result"] = result
    st.session_state.pop("pending_graph", None)
    st.session_state.pop("pending_config", None)
    st.rerun()


def _run_single_demo(checkpointer_kind: str) -> None:
    left, right = st.columns([1.5, 1])
    with left:
        preset_name = st.selectbox("Demo preset", list(PRESETS))
        preset = PRESETS[preset_name]
        query = st.text_area(
            "Support ticket",
            value=str(preset["query"]),
            height=110,
            key=f"query-{preset_name}",
        )
    with right:
        routes = [route.value for route in Route if route not in {Route.DEAD_LETTER, Route.DONE}]
        preset_route = cast(Route, preset["route"]).value
        expected_route = st.selectbox(
            "Expected route",
            routes,
            index=routes.index(preset_route),
            key=f"route-{preset_name}",
        )
        preset_max_attempts = preset["max_attempts"]
        max_attempts = st.number_input(
            "Max attempts",
            min_value=1,
            max_value=5,
            value=preset_max_attempts if isinstance(preset_max_attempts, int) else 3,
        )
        real_hitl = st.toggle(
            "Real HITL interrupt",
            value=preset_route == Route.RISKY.value,
            help="Risky routes pause at approval and resume from the same checkpoint.",
        )

    run_clicked = st.button("Run workflow", type="primary", width="stretch")
    if run_clicked:
        if not query.strip():
            st.error("Enter a support ticket before running the graph.")
        else:
            scenario = Scenario(
                id=f"ui-{uuid4().hex[:8]}",
                query=query,
                expected_route=Route(expected_route),
                requires_approval=expected_route == Route.RISKY.value,
                max_attempts=int(max_attempts),
            )
            state = initial_state(scenario)
            graph = _cached_graph(checkpointer_kind)
            config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
            with st.spinner("Invoking LangGraph and the configured LLM..."):
                result = _invoke(graph, state, config, real_hitl=real_hitl)
            st.session_state["last_result"] = result
            st.session_state["last_expected_route"] = expected_route
            st.session_state["last_config"] = config
            st.session_state["last_graph"] = graph
            if _has_interrupt(result):
                st.session_state["pending_graph"] = graph
                st.session_state["pending_config"] = config
            else:
                st.session_state.pop("pending_graph", None)
                st.session_state.pop("pending_config", None)

    _render_pending_approval()
    if "last_result" in st.session_state:
        _render_result(
            cast(AgentState, st.session_state["last_result"]),
            str(st.session_state.get("last_expected_route", expected_route)),
        )


def _run_scenario_suite(checkpointer_kind: str) -> None:
    st.write(
        "Run the seven grading scenarios with real LLM classification and grounded answers. "
        "Risky actions use deterministic mock approval so the suite can finish unattended."
    )
    if st.button("Run all sample scenarios", type="primary"):
        scenarios = load_scenarios(SAMPLE_SCENARIOS)
        graph = _cached_graph(checkpointer_kind)
        progress = st.progress(0, text="Starting scenario suite...")
        items: list[ScenarioMetric] = []
        history_observed = False
        for index, scenario in enumerate(scenarios, start=1):
            state = initial_state(scenario)
            state["thread_id"] = f"ui-suite-{scenario.id}-{uuid4().hex[:6]}"
            config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
            result = _invoke(graph, state, config, real_hitl=False)
            items.append(
                metric_from_state(
                    cast(dict[str, object], result),
                    scenario.expected_route.value,
                    scenario.requires_approval,
                )
            )
            if checkpointer_kind == "sqlite":
                history_observed = history_observed or bool(
                    next(graph.get_state_history(config), None)
                )
            progress.progress(index / len(scenarios), text=f"Completed {scenario.id}")
        report = summarize_metrics(items, resume_success=history_observed)
        st.session_state["suite_report"] = report
        progress.empty()

    if "suite_report" not in st.session_state:
        st.caption("No suite has been run in this UI session yet.")
        return

    report = cast(MetricsReport, st.session_state["suite_report"])
    cols = st.columns(6)
    cols[0].metric("Success", f"{report.success_rate:.0%}")
    cols[1].metric("Scenarios", report.total_scenarios)
    cols[2].metric("Retries", report.total_retries)
    cols[3].metric("Approvals", report.total_interrupts)
    cols[4].metric("LLM calls", report.total_llm_calls)
    cols[5].metric("Fallbacks", report.total_structured_fallbacks)

    rows = [item.model_dump() for item in report.scenario_metrics]
    st.dataframe(rows, width="stretch", hide_index=True)

    report_markdown = render_report(report)
    metrics_json = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download metrics.json",
        metrics_json,
        file_name="metrics.json",
        mime="application/json",
        width="stretch",
    )
    download_cols[1].download_button(
        "Download report.md",
        report_markdown,
        file_name="lab_report.md",
        mime="text/markdown",
        width="stretch",
    )
    if download_cols[2].button("Save into repository", width="stretch"):
        write_metrics(report, UI_METRICS)
        write_report(report, UI_REPORT)
        st.success(f"Saved {UI_METRICS.name} and {UI_REPORT.name}")

    with st.expander("Preview generated report"):
        st.markdown(report_markdown)


def _render_trace_and_checkpoints() -> None:
    if "last_result" not in st.session_state:
        st.info("Run a workflow in the Live Demo tab to populate trace and checkpoint history.")
        return
    state = cast(AgentState, st.session_state["last_result"])
    graph = cast(CompiledStateGraph, st.session_state["last_graph"])
    config = cast(RunnableConfig, st.session_state["last_config"])

    st.subheader("Execution replay")
    path = _execution_path(state)
    slider_col, replay_col = st.columns([4, 1])
    replay_step = slider_col.slider(
        "Workflow step",
        min_value=0,
        max_value=len(path) - 1,
        value=len(path) - 1,
        format="Step %d",
        key=f"replay-{state.get('thread_id', 'unknown')}",
    )
    replay_clicked = replay_col.button("Replay animation", width="stretch")
    graph_placeholder = st.empty()
    caption_placeholder = st.empty()
    if replay_clicked:
        for step in range(len(path)):
            graph_placeholder.graphviz_chart(
                _dynamic_graph_dot(state, upto_step=step), width="stretch"
            )
            caption_placeholder.caption(f"Step {step}: {_path_caption(path, step)}")
            sleep(0.45)
    else:
        graph_placeholder.graphviz_chart(
            _dynamic_graph_dot(state, upto_step=replay_step), width="stretch"
        )
        caption_placeholder.caption(f"Step {replay_step}: {_path_caption(path, replay_step)}")

    st.subheader("Node-by-node trace")
    st.dataframe(_event_rows(state), width="stretch", hide_index=True)

    st.subheader("Checkpoint history")
    st.caption(f"Thread ID: {state.get('thread_id', 'unknown')}")
    history = _history_rows(graph, config)
    if history:
        st.dataframe(history, width="stretch", hide_index=True)
        st.success(f"Recovered {len(history)} checkpoint snapshots from the active thread.")
    else:
        st.warning("No checkpoint history is available for this run.")

    st.subheader("Append-only audit evidence")
    audit_cols = st.columns(4)
    audit_cols[0].metric("Events", len(state.get("events", [])))
    audit_cols[1].metric("Messages", len(state.get("messages", [])))
    audit_cols[2].metric("Tool results", len(state.get("tool_results", [])))
    audit_cols[3].metric("Errors", len(state.get("errors", [])))


def _render_architecture() -> None:
    st.graphviz_chart(GRAPH_DOT, width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Conditional routing")
        st.dataframe(
            [
                {"route": "simple", "path": "answer → finalize"},
                {"route": "tool", "path": "tool → evaluate → answer/retry"},
                {"route": "missing_info", "path": "clarify → finalize"},
                {"route": "risky", "path": "risky_action → approval → tool/clarify"},
                {"route": "error", "path": "retry → tool/dead_letter"},
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("State reducers")
        st.dataframe(
            [
                {"fields": "messages, tool_results, errors, events", "reducer": "append"},
                {
                    "fields": "route, attempt, evaluation, approval, final_answer",
                    "reducer": "overwrite",
                },
            ],
            width="stretch",
            hide_index=True,
        )
    st.markdown(
        """
        <span class="pill">Structured classification</span>
        <span class="pill">Bounded retry</span>
        <span class="pill">Human-in-the-loop</span>
        <span class="pill">SQLite persistence</span>
        <span class="pill">LLM retry + fallback</span>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the complete live-demo dashboard."""
    st.set_page_config(
        page_title="LangGraph Agent Lab",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    st.markdown(
        """
        <div class="hero">
          <h1>LangGraph Support Agent Lab</h1>
          <p>Conditional routing · retry loops · HITL · persistence · metrics · tracing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Demo controls")
        checkpointer_kind = st.radio(
            "Checkpointer",
            ["sqlite", "memory"],
            horizontal=True,
            help="SQLite demonstrates durable checkpoint history across UI reruns.",
        )
        model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "default"
        endpoint = os.getenv("GEMINI_BASE_URL")
        endpoint_host = urlparse(endpoint).netloc if endpoint else "native provider"
        key_ready = bool(
            os.getenv("GEMINI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        st.divider()
        st.caption("LLM configuration")
        st.write(f"**Model:** `{model}`")
        st.write(f"**Endpoint:** `{endpoint_host}`")
        st.write("**API key:** configured" if key_ready else "**API key:** missing")
        st.caption("The key value is never rendered in the dashboard.")

    demo_tab, suite_tab, trace_tab, architecture_tab = st.tabs(
        ["▶ Live Demo", "▦ Scenario Suite", "⌁ Trace & Checkpoints", "◇ Architecture"]
    )
    with demo_tab:
        _run_single_demo(checkpointer_kind)
    with suite_tab:
        _run_scenario_suite(checkpointer_kind)
    with trace_tab:
        _render_trace_and_checkpoints()
    with architecture_tab:
        _render_architecture()


if __name__ == "__main__":
    main()
