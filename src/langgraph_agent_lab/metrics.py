"""Metrics schema and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field


class ScenarioMetric(BaseModel):
    scenario_id: str
    success: bool
    expected_route: str
    actual_route: str | None = None
    nodes_visited: int = 0
    retry_count: int = 0
    interrupt_count: int = 0
    approval_required: bool = False
    approval_observed: bool = False
    latency_ms: int = 0
    llm_calls: int = 0
    llm_judge_calls: int = 0
    structured_fallback_count: int = 0
    errors: list[str] = Field(default_factory=list)


class MetricsReport(BaseModel):
    total_scenarios: int
    success_rate: float
    avg_nodes_visited: float
    total_retries: int
    total_interrupts: int
    total_llm_calls: int = 0
    total_llm_judge_calls: int = 0
    total_structured_fallbacks: int = 0
    resume_success: bool = False
    scenario_metrics: list[ScenarioMetric]


def metric_from_state(
    state: dict[str, Any], expected_route: str, approval_required: bool
) -> ScenarioMetric:
    events = state.get("events", []) or []
    errors = state.get("errors", []) or []
    actual_route = state.get("route")
    approval = state.get("approval")
    nodes = [event.get("node", "unknown") for event in events]
    latency_ms = sum(int(event.get("latency_ms", 0) or 0) for event in events)
    event_metadata = [event.get("metadata", {}) or {} for event in events]
    llm_calls = sum(int(metadata.get("llm_calls", 0) or 0) for metadata in event_metadata)
    llm_judge_calls = sum(
        int(metadata.get("llm_calls", 0) or 0)
        for metadata in event_metadata
        if metadata.get("evaluation_mode") == "llm_judge"
    )
    structured_fallback_count = sum(
        1 for metadata in event_metadata if metadata.get("structured_fallback") is True
    )
    retry_count = sum(1 for node in nodes if node == "retry")
    interrupt_count = sum(1 for node in nodes if node == "approval")
    dead_letter_observed = "dead_letter" in nodes
    success = actual_route == expected_route and bool(
        state.get("final_answer") or state.get("pending_question")
    )
    if approval_required:
        success = success and approval is not None
    if expected_route == "error":
        max_attempts = int(state.get("max_attempts", 3) or 3)
        expected_dead_letter = max_attempts <= 1
        success = success and dead_letter_observed is expected_dead_letter
    return ScenarioMetric(
        scenario_id=str(state.get("scenario_id", "unknown")),
        success=success,
        expected_route=expected_route,
        actual_route=actual_route,
        nodes_visited=len(nodes),
        retry_count=retry_count,
        interrupt_count=interrupt_count,
        approval_required=approval_required,
        approval_observed=approval is not None,
        latency_ms=latency_ms,
        llm_calls=llm_calls,
        llm_judge_calls=llm_judge_calls,
        structured_fallback_count=structured_fallback_count,
        errors=list(errors),
    )


def summarize_metrics(
    items: list[ScenarioMetric], *, resume_success: bool = False
) -> MetricsReport:
    if not items:
        raise ValueError("No scenario metrics to summarize")
    return MetricsReport(
        total_scenarios=len(items),
        success_rate=sum(1 for item in items if item.success) / len(items),
        avg_nodes_visited=mean(item.nodes_visited for item in items),
        total_retries=sum(item.retry_count for item in items),
        total_interrupts=sum(item.interrupt_count for item in items),
        total_llm_calls=sum(item.llm_calls for item in items),
        total_llm_judge_calls=sum(item.llm_judge_calls for item in items),
        total_structured_fallbacks=sum(item.structured_fallback_count for item in items),
        resume_success=resume_success,
        scenario_metrics=items,
    )


def write_metrics(report: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
