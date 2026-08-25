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


def test_evaluate_explicit_error_skips_llm(monkeypatch):
    def fail_if_called(temperature=0):
        raise AssertionError("explicit tool outcome should not call the LLM")

    monkeypatch.setattr(nodes, "get_llm", fail_if_called)
    result = nodes.evaluate_node({"tool_results": ["ERROR: timeout"]})

    assert result["evaluation_result"] == "needs_retry"
    metadata = result["events"][0]["metadata"]
    assert metadata["evaluation_mode"] == "heuristic"
    assert metadata["llm_calls"] == 0


def test_success_status_wins_over_error_words_in_payload(monkeypatch):
    def fail_if_called(temperature=0):
        raise AssertionError("explicit success status should not call the LLM")

    monkeypatch.setattr(nodes, "get_llm", fail_if_called)
    result = nodes.evaluate_node({"tool_results": ["SUCCESS: recovered from timeout failure"]})

    assert result["evaluation_result"] == "success"


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
