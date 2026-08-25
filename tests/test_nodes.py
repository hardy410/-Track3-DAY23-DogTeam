"""Fast unit tests for optimized LLM and evaluation paths."""

from langchain_core.messages import AIMessage
from tenacity import wait_none

from langgraph_agent_lab import llm, nodes


class BrokenStructuredRunnable:
    def invoke(self, prompt):
        raise ValueError("structured mode unsupported")


class JsonFallbackLLM:
    def with_structured_output(self, schema, method):
        assert method == "function_calling"
        return BrokenStructuredRunnable()

    def invoke(self, prompt):
        return AIMessage(content='{"route":"simple","reasoning":"general guidance"}')


class ToolJsonFallbackLLM:
    def with_structured_output(self, schema, method):
        return BrokenStructuredRunnable()

    def invoke(self, prompt):
        return AIMessage(content='{"route":"tool","reasoning":"order tracking request"}')


class JudgeRunnable:
    def invoke(self, prompt):
        return nodes.EvaluationResult(
            evaluation_result="success",
            reasoning="result is usable",
        )


class JudgeLLM:
    def with_structured_output(self, schema, method):
        assert method == "function_calling"
        return JudgeRunnable()


class FlakyRunnable:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary network failure")
        return "ok"


def test_classify_uses_validated_json_fallback(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0: JsonFallbackLLM())

    result = nodes.classify_node({"query": "How do I reset my password?"})

    assert result["route"] == "simple"
    metadata = result["events"][0]["metadata"]
    assert metadata["structured_fallback"] is True
    assert metadata["llm_calls"] == 2


def test_classify_requires_order_id_before_tool_lookup(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0: ToolJsonFallbackLLM())

    result = nodes.classify_node(
        {"query": "Đơn mình đặt vẫn chưa thấy giao tới. Bạn kiểm tra giúp mình nhé?"}
    )

    assert result["route"] == "missing_info"
    metadata = result["events"][0]["metadata"]
    assert metadata["llm_route"] == "tool"
    assert metadata["policy_override"] is True


def test_evaluate_explicit_error_skips_llm(monkeypatch):
    def fail_if_called(temperature=0):
        raise AssertionError("explicit tool outcome should not call the LLM")

    monkeypatch.setattr(nodes, "get_llm", fail_if_called)
    result = nodes.evaluate_node({"tool_results": ["ERROR: timeout"]})

    assert result["evaluation_result"] == "needs_retry"
    metadata = result["events"][0]["metadata"]
    assert metadata["evaluation_mode"] == "heuristic"
    assert metadata["llm_calls"] == 0


def test_missing_order_id_routes_to_clarification_without_retry(monkeypatch):
    def fail_if_called(temperature=0):
        raise AssertionError("explicit missing info should not call the LLM judge")

    monkeypatch.setattr(nodes, "get_llm", fail_if_called)
    tool_result = nodes.tool_node(
        {
            "route": "tool",
            "query": "Đơn mình vẫn chưa thấy giao, kiểm tra giúp mình nhé?",
            "should_retry": False,
            "attempt": 0,
        }
    )
    evaluated = nodes.evaluate_node({"tool_results": tool_result["tool_results"]})

    assert tool_result["tool_results"][0].startswith("NEEDS_INFO:")
    assert evaluated["evaluation_result"] == "missing_info"


def test_success_status_wins_over_error_words_in_payload(monkeypatch):
    def fail_if_called(temperature=0):
        raise AssertionError("explicit success status should not call the LLM")

    monkeypatch.setattr(nodes, "get_llm", fail_if_called)
    result = nodes.evaluate_node({"tool_results": ["SUCCESS: recovered from timeout failure"]})

    assert result["evaluation_result"] == "success"


def test_tool_retry_failure_is_independent_from_customer_intent():
    first_call = nodes.tool_node(
        {
            "route": "tool",
            "query": "Tra cứu đơn #12345",
            "should_retry": True,
            "attempt": 0,
        }
    )
    recovered_call = nodes.tool_node(
        {
            "route": "tool",
            "query": "Tra cứu đơn #12345",
            "should_retry": True,
            "attempt": 2,
        }
    )

    assert first_call["tool_results"][0].startswith("ERROR: lần gọi công cụ 1 bị timeout")
    recovered_result = recovered_call["tool_results"][0]
    assert recovered_result.startswith("SUCCESS: Đã khớp chính xác khóa #12345")
    assert "bản ghi ORD-12345-HN" in recovered_result
    assert "Vị trí hiện tại: Bưu cục Cầu Giấy, Hà Nội" in recovered_result
    assert "Dự kiến giao: Trước 18:00 ngày 25/08/2026" in recovered_result


def test_tool_returns_clear_mock_tracking_data():
    result = nodes.tool_node(
        {
            "route": "tool",
            "query": "Đơn #12345 đang được giao tới đâu?",
            "should_retry": False,
            "attempt": 0,
        }
    )

    tool_result = result["tool_results"][0]
    assert "Trạng thái: Đang giao hàng" in tool_result
    assert "Vị trí hiện tại: Bưu cục Cầu Giấy, Hà Nội" in tool_result
    assert "Cập nhật lúc: 15:30 ngày 25/08/2026" in tool_result
    assert "Tuyến giao: Kho Long Biên, Hà Nội → Phường Dịch Vọng" in tool_result
    assert "Đã thanh toán 450.000 đồng bằng thẻ Visa •••• 0410" in tool_result
    assert "Tài xế sẽ gọi người nhận trước khi giao" in tool_result


def test_order_lookup_selects_the_requested_record_not_any_available_record():
    requested = nodes.tool_node(
        {
            "route": "tool",
            "query": "Kiểm tra đơn #12345",
            "should_retry": False,
            "attempt": 0,
        }
    )["tool_results"][0]
    other = nodes.tool_node(
        {
            "route": "tool",
            "query": "Kiểm tra đơn #67890",
            "should_retry": False,
            "attempt": 0,
        }
    )["tool_results"][0]

    assert "ORD-12345-HN" in requested
    assert "Đang giao hàng" in requested
    assert "ORD-67890-HCM" not in requested
    assert "ORD-67890-HCM" in other
    assert "Đã giao thành công" in other
    assert "ORD-12345-HN" not in other


def test_unknown_order_does_not_fall_back_to_an_arbitrary_record():
    result = nodes.tool_node(
        {
            "route": "tool",
            "query": "Kiểm tra đơn #99999",
            "should_retry": False,
            "attempt": 0,
        }
    )

    assert result["tool_results"] == [
        "NOT_FOUND: không tìm thấy đơn hàng #99999 trong mock_orders.json"
    ]
    assert nodes.evaluate_node(result)["evaluation_result"] == "success"


def test_evaluate_ambiguous_result_uses_judge(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0: JudgeLLM())
    result = nodes.evaluate_node({"tool_results": ["Retrieved a payload"]})

    assert result["evaluation_result"] == "success"
    metadata = result["events"][0]["metadata"]
    assert metadata["evaluation_mode"] == "llm_judge"
    assert metadata["llm_calls"] == 1


def test_transient_llm_failure_is_retried(monkeypatch):
    monkeypatch.setattr(llm, "wait_exponential", lambda **kwargs: wait_none())
    runnable = FlakyRunnable()

    result, attempts = llm.invoke_with_retry(runnable, "hello")

    assert result == "ok"
    assert attempts == 2


def test_non_transient_llm_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(llm, "wait_exponential", lambda **kwargs: wait_none())

    class InvalidRequestRunnable:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            raise ValueError("invalid request")

    runnable = InvalidRequestRunnable()
    try:
        llm.invoke_with_retry(runnable, "hello")
    except ValueError:
        pass
    else:
        raise AssertionError("non-transient failure should be raised")

    assert runnable.calls == 1


def test_wait_for_user_merges_reply_into_same_request(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_CLARIFY_INTERRUPT", "true")
    monkeypatch.setattr(
        "langgraph.types.interrupt",
        lambda payload: {"answer": "Mã đơn là #12345, mình muốn biết đang giao tới đâu."},
    )

    result = nodes.wait_for_user_node(
        {
            "query": "Đơn hàng của mình có vấn đề, bạn xử lý giúp mình được không?",
            "pending_question": "Bạn cho mình xin mã đơn hàng nhé?",
        }
    )

    assert result["clarification_received"] is True
    assert "#12345" in result["query"]
    assert result["pending_question"] is None
    assert result["events"][0]["node"] == "wait_for_user"
