"""Streamlit dashboard smoke tests with a safe deterministic LLM."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from langgraph_agent_lab import nodes, ui
from langgraph_agent_lab.state import make_event

UI_PATH = Path(__file__).parents[1] / "src" / "langgraph_agent_lab" / "ui.py"


class FakeStructuredRunnable:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        if self.schema is nodes.ClassificationResult:
            return self.schema(route="risky", reasoning="side-effecting request")
        return self.schema(evaluation_result="success", reasoning="usable result")


class FakeSupportLLM:
    def with_structured_output(self, schema, method):
        assert method == "function_calling"
        return FakeStructuredRunnable(schema)

    def invoke(self, prompt):
        return AIMessage(content="The approved mock action completed successfully.")


def test_dashboard_renders_without_exceptions():
    app = AppTest.from_file(UI_PATH, default_timeout=30).run()

    assert not app.exception
    assert len(app.tabs) == 4
    assert any(button.label == "Run workflow" for button in app.button)


def test_dynamic_graph_highlights_executed_retry_path():
    state = {
        "events": [
            make_event("intake", "completed", "ok"),
            make_event("classify", "completed", "error"),
            make_event("retry", "scheduled", "retry"),
            make_event("dead_letter", "failed", "exhausted"),
            make_event("finalize", "completed", "done"),
        ]
    }

    path = ui._execution_path(state)
    graph = ui._dynamic_graph_dot(state)

    assert path == [
        "START",
        "intake",
        "classify",
        "retry",
        "dead_letter",
        "finalize",
        "END",
    ]
    assert 'classify -> retry [color="#D92D20", penwidth=3' in graph
    assert 'retry -> dead_letter [color="#D92D20", penwidth=3' in graph


def test_risky_workflow_pauses_and_resumes_with_hitl(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0: FakeSupportLLM())
    app = AppTest.from_file(UI_PATH, default_timeout=30).run()
    app.selectbox[0].select("Risky refund + HITL").run()
    app.toggle[0].set_value(True).run()

    run_button = next(button for button in app.button if button.label == "Run workflow")
    run_button.click().run(timeout=30)

    assert not app.exception
    assert any(button.label == "Approve" for button in app.button)
    assert app.warning

    approve_button = next(button for button in app.button if button.label == "Approve")
    approve_button.click().run(timeout=30)

    assert not app.exception
    assert app.success
    assert any(metric.label == "Actual route" and metric.value == "risky" for metric in app.metric)
    assert len(app.metric) >= 6
    assert len(app.dataframe) >= 1
