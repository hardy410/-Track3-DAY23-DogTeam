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


class ConversationalStructuredRunnable:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        if self.schema is nodes.ClassificationResult:
            route = "tool" if "Thông tin người dùng bổ sung" in prompt else "missing_info"
            return self.schema(route=route, reasoning="conversation has enough context")
        return self.schema(evaluation_result="success", reasoning="usable result")


class ConversationalSupportLLM:
    def with_structured_output(self, schema, method):
        assert method == "function_calling"
        return ConversationalStructuredRunnable(schema)

    def invoke(self, prompt):
        return AIMessage(content="Đơn #12345 đang được giao và dự kiến đến trong hôm nay.")


def test_dashboard_renders_without_exceptions():
    app = AppTest.from_file(UI_PATH, default_timeout=30).run()

    assert not app.exception
    assert ui.PIXEL_WORLD_VIDEO.exists()
    assert ui.PIXEL_WORLD_IMAGE.exists()
    assert any("world-video" in item.value for item in app.markdown)
    assert app.session_state["chat_max_attempts"] == 3
    assert app.session_state["show_live_graph"] is True
    assert app.session_state["chat_hitl"] is True
    assert len(app.toggle) == 0
    assert len(app.tabs) == 4
    assert any(button.label == "＋  NHIỆM VỤ MỚI" for button in app.button)
    assert any(button.label == "◇  Phê duyệt hoàn tiền" for button in app.button)


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
    full_graph = ui._full_graph_map_html(state)

    assert path == [
        "START",
        "intake",
        "classify",
        "retry",
        "dead_letter",
        "finalize",
        "END",
    ]
    assert 'classify -> retry [color="#FF806F", penwidth=3' in graph
    assert 'retry -> dead_letter [color="#FF806F", penwidth=3' in graph
    assert len(full_graph.split('class="graph-node')) == len(ui.GRAPH_NODES) + 1
    assert "graph-node danger visited" in full_graph
    assert "graph-node terminal visited current" in full_graph
    assert "Chuyển nhân viên" in full_graph


def test_new_conversation_keeps_multiple_chat_threads():
    app = AppTest.from_file(UI_PATH, default_timeout=30).run()
    new_chat = next(button for button in app.button if button.label == "＋  NHIỆM VỤ MỚI")
    new_chat.click().run()

    assert not app.exception
    assert len(app.session_state["conversations"]) == 2


def test_risky_workflow_pauses_and_resumes_with_hitl(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0: FakeSupportLLM())
    app = AppTest.from_file(UI_PATH, default_timeout=30).run()

    run_button = next(button for button in app.button if button.label == "◇  Phê duyệt hoàn tiền")
    run_button.click().run(timeout=30)

    assert not app.exception
    assert any("Toàn bộ graph" in item.value for item in app.markdown)
    assert any("HIỂU Ý ĐỊNH" in item.value for item in app.markdown)
    assert any(button.label == "✓ PHÊ DUYỆT & TIẾP TỤC" for button in app.button)
    assert app.warning

    approve_button = next(
        button for button in app.button if button.label == "✓ PHÊ DUYỆT & TIẾP TỤC"
    )
    approve_button.click().run(timeout=30)

    assert not app.exception
    return_button = next(button for button in app.button if button.label == "VỀ CUỘC HỘI THOẠI")
    return_button.click().run(timeout=30)

    assert not app.exception
    assert any("approved mock action" in item.value.lower() for item in app.markdown)
    assert len(app.dataframe) >= 1


def test_rejected_risky_action_asks_user_instead_of_ending(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0: FakeSupportLLM())
    app = AppTest.from_file(UI_PATH, default_timeout=30).run()
    run_button = next(button for button in app.button if button.label == "◇  Phê duyệt hoàn tiền")
    run_button.click().run(timeout=30)

    reject_button = next(button for button in app.button if button.label == "✕ TỪ CHỐI")
    reject_button.click().run(timeout=30)

    assert not app.exception
    assert any(button.label == "ẨN GRAPH & TRẢ LỜI TRONG CHAT" for button in app.button)
    turns = next(iter(app.session_state["conversations"].values()))["turns"]
    state = turns[0]["state"]
    assert ui._is_clarification_interrupt(state)
    assert state["pending_question"]
    assert "END" not in ui._execution_path(state)


def test_clarification_reply_resumes_same_graph_and_reclassifies(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0: ConversationalSupportLLM())
    app = AppTest.from_file(UI_PATH, default_timeout=30).run()

    shortcut = next(button for button in app.button if button.label == "?  Yêu cầu mơ hồ")
    shortcut.click().run(timeout=30)
    close_graph = next(
        button for button in app.button if button.label == "ẨN GRAPH & TRẢ LỜI TRONG CHAT"
    )
    close_graph.click().run(timeout=30)

    assert not app.exception
    assert any("mã đơn hàng" in item.value.lower() for item in app.markdown)
    assert not any(button.label == "Phê duyệt & tiếp tục" for button in app.button)
    assert not any(button.label == "Từ chối" for button in app.button)
    app.chat_input[0].set_value("Mã đơn là #12345, mình muốn biết đang giao tới đâu.").run(
        timeout=30
    )

    assert not app.exception
    return_button = next(button for button in app.button if button.label == "VỀ CUỘC HỘI THOẠI")
    return_button.click().run(timeout=30)
    turns = next(iter(app.session_state["conversations"].values()))["turns"]
    assert len(turns) == 1
    assert turns[0]["state"]["route"] == "tool"
    assert turns[0]["clarifications"][0]["answer"].startswith("Mã đơn là #12345")
