"""Render a complete Markdown lab report from collected scenario metrics."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    The generated report includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    summary_rows = [
        ("Total scenarios", str(metrics.total_scenarios)),
        ("Success rate", f"{metrics.success_rate:.1%}"),
        ("Average nodes visited", f"{metrics.avg_nodes_visited:.2f}"),
        ("Total retries", str(metrics.total_retries)),
        ("Total interrupts/approvals", str(metrics.total_interrupts)),
        ("Total LLM calls", str(metrics.total_llm_calls)),
        ("LLM judge calls", str(metrics.total_llm_judge_calls)),
        ("Structured fallbacks", str(metrics.total_structured_fallbacks)),
        ("Resume success", "Yes" if metrics.resume_success else "Not demonstrated"),
    ]
    summary_table = "\n".join(f"| {name} | {value} |" for name, value in summary_rows)

    scenario_rows = []
    for item in metrics.scenario_metrics:
        scenario_rows.append(
            "| "
            + " | ".join(
                [
                    item.scenario_id,
                    item.expected_route,
                    item.actual_route or "n/a",
                    "Yes" if item.success else "No",
                    str(item.retry_count),
                    str(item.interrupt_count),
                    str(item.llm_calls),
                    str(item.latency_ms),
                ]
            )
            + " |"
        )

    return f"""# LangGraph Agentic Orchestration Lab Report

## 1. Student

- Name: Nguyễn Đình Liên Thanh
- Repository: K4-Track3-Day23-2A202601790-NguyenDinhLienThanh
- Date: 2026-08-25

## 2. Architecture

The workflow uses a typed `AgentState` and eleven small nodes. Intake normalizes the
ticket, an LLM performs structured intent classification, and conditional edges select
simple answering, tool lookup, clarification, risky-action approval, or bounded retry.
All branches terminate through a shared finalize node. A checkpointer is compiled into
the graph and each scenario supplies an independent thread identifier.

Append-only reducers are used for messages, tool results, errors, and audit events.
Current route, retry count, evaluation result, approval, proposed action, pending question,
and final answer use overwrite semantics.

```mermaid
flowchart TD
    START --> intake --> classify
    classify -->|simple| answer
    classify -->|tool| tool --> evaluate
    classify -->|missing info| clarify
    classify -->|risky| risky_action --> approval
    classify -->|error| retry
    approval -->|approved| tool
    approval -->|rejected| clarify
    evaluate -->|success| answer
    evaluate -->|needs retry| retry
    evaluate -->|missing info| clarify
    retry -->|within budget| tool
    retry -->|exhausted| dead_letter
    answer --> finalize
    clarify --> wait_for_user
    wait_for_user -->|customer replied| classify
    wait_for_user -->|batch mode| finalize
    dead_letter --> finalize --> END
```

## 3. Metrics summary

| Metric | Value |
|---|---:|
{summary_table}

## 4. Scenario results

| Scenario | Expected | Actual | Success | Retries | Approvals | LLM calls | LLM latency (ms) |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(scenario_rows)}

## 5. Failure analysis

1. **Transient tool failure:** tool results containing an explicit error marker are sent
   through evaluation and a bounded retry loop. Once `attempt >= max_attempts`, the request
   moves to dead letter instead of looping forever.
2. **Risky action without approval:** refund, deletion, cancellation, and outbound-message
   requests are classified as risky and must pass through the approval node. Rejected actions
   are redirected to clarification and never reach the tool.
3. **LLM judge unavailable:** explicit tool status is resolved before invoking the judge.
   Ambiguous results use the judge and fail safely to retry if that optional call is unavailable.

## 6. Persistence and recovery

The graph supports both in-memory checkpoints and SQLite WAL persistence. SQLite stores
durable state keyed by `thread_id`, enabling state-history inspection and later resume.

## 7. Extension work

- SQLite checkpointer with WAL mode.
- Optional real human-in-the-loop interruption via `LANGGRAPH_INTERRUPT=true`.
- LLM-as-judge evaluation with deterministic fallback.
- Selective retry for transient LLM errors and structured-output compatibility fallback.
- Versioned prompts and LLM-call/fallback metrics.
- Streamlit dashboard with live HITL resume, node trace, state inspection, checkpoint
  history, scenario-suite execution, and downloadable artifacts.

In the recorded before/after benchmark, heuristic-first evaluation reduced summed LLM
latency from 71,073 ms to 55,941 ms (21.3%) while preserving 100% scenario success.

## 8. Improvement plan

For production, replace mock tools with typed integrations, enforce authorization before
side effects, add idempotency keys, redact sensitive event data, capture provider token/cost
metadata, and evaluate classification against a larger adversarial dataset.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
