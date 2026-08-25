"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Literal, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ValidationError

from .llm import get_llm, invoke_with_retry
from .mock_tools import (
    extract_order_id,
    format_tracking_result,
    is_order_lookup_without_id,
    lookup_order,
)
from .prompts import PROMPT_VERSION, answer_prompt, classification_prompt, evaluation_prompt
from .state import AgentState, ApprovalDecision, Route, make_event

ModelT = TypeVar("ModelT", bound=BaseModel)


class ClassificationResult(BaseModel):
    """Validated output returned by the intent classifier."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    reasoning: str = Field(description="Brief reason for selecting the route")


class EvaluationResult(BaseModel):
    """Validated LLM-as-judge output for a mock tool result."""

    evaluation_result: Literal["success", "needs_retry", "missing_info"]
    reasoning: str = Field(description="Brief quality assessment")


def _message_text(message: BaseMessage) -> str:
    """Normalize text-only and multimodal LangChain message content."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts).strip()


def _parse_json_model(schema: type[ModelT], text: str) -> ModelT:
    """Parse a JSON object even when a model wraps it in Markdown prose."""
    stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return schema.model_validate_json(stripped[start : end + 1])


def _is_structured_compatibility_error(exc: Exception) -> bool:
    """Identify schema/tool incompatibility without masking auth or server failures."""
    status_code = getattr(exc, "status_code", None)
    return isinstance(exc, (OutputParserException, ValidationError, ValueError)) or status_code in {
        400,
        422,
    }


def _invoke_structured(schema: type[ModelT], prompt: str) -> tuple[ModelT, int, bool]:
    """Prefer function calling, with a validated JSON-prompt compatibility fallback."""
    llm = get_llm(temperature=0)
    try:
        structured = llm.with_structured_output(schema, method="function_calling")
        result, calls = invoke_with_retry(structured, prompt)
        validated = result if isinstance(result, schema) else schema.model_validate(result)
        return validated, calls, False
    except Exception as exc:
        if not _is_structured_compatibility_error(exc):
            raise

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    fallback_prompt = (
        f"{prompt}\nReturn only one valid JSON object matching this schema:\n{schema_json}"
    )
    raw_response, calls = invoke_with_retry(llm, fallback_prompt)
    return _parse_json_model(schema, _message_text(raw_response)), calls + 1, True


def _heuristic_tool_evaluation(
    tool_result: str,
) -> Literal["success", "needs_retry", "missing_info"] | None:
    """Resolve explicit mock-tool outcomes without spending an LLM call."""
    normalized = tool_result.strip().upper()
    if normalized.startswith("NEEDS_INFO:"):
        return "missing_info"
    if normalized.startswith("NOT_FOUND:"):
        return "success"
    if normalized.startswith("SUCCESS:"):
        return "success"
    if normalized.startswith("ERROR:"):
        return "needs_retry"
    error_markers = ("ERROR", "FAILED", "FAILURE", "TIMEOUT", "UNAVAILABLE")
    success_markers = ("SUCCESS", "COMPLETED", "OK")
    if any(marker in normalized for marker in error_markers):
        return "needs_retry"
    if any(marker in normalized for marker in success_markers):
        return "success"
    return None


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "Đã chuẩn hóa yêu cầu đầu vào")],
    }


# ─── Workflow node implementations ───────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    prompt = classification_prompt(state.get("query", ""))
    started = perf_counter()
    result, llm_calls, structured_fallback = _invoke_structured(ClassificationResult, prompt)
    latency_ms = int((perf_counter() - started) * 1000)
    llm_route = result.route
    route = result.route
    policy_override = False
    reasoning = result.reasoning
    if route == Route.TOOL.value and is_order_lookup_without_id(state.get("query", "")):
        route = Route.MISSING_INFO.value
        policy_override = True
        reasoning = f"{reasoning}; cần mã đơn hàng trước khi gọi công cụ"
    risk_level = "high" if route == Route.RISKY.value else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"Đã phân loại vào route {route}",
                latency_ms=latency_ms,
                reasoning=reasoning,
                llm_route=llm_route,
                policy_override=policy_override,
                llm_calls=llm_calls,
                prompt_version=PROMPT_VERSION,
                structured_fallback=structured_fallback,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient tool failures independently from the customer's intent.

    Requirements:
    - Read current attempt count from state
    - If the scenario enables retry and attempt < 2: return an explicit timeout
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")
    should_retry = state.get("should_retry", route == Route.ERROR.value)
    call_number = attempt + 1
    if should_retry and attempt < 2:
        result = (
            f"ERROR: lần gọi công cụ {call_number} bị timeout; "
            "dịch vụ vận chuyển chưa phản hồi"
        )
        event_type = "failed"
    elif route == Route.RISKY.value:
        action = state.get("proposed_action") or query
        result = f"SUCCESS: hành động đã duyệt được thực hiện an toàn: {action}"
        event_type = "completed"
    else:
        order = lookup_order(query)
        if order is None:
            requested_id = extract_order_id(query)
            result = (
                f"NOT_FOUND: không tìm thấy đơn hàng #{requested_id} trong mock_orders.json"
                if requested_id
                else "NEEDS_INFO: cần mã đơn hàng để thực hiện tra cứu chính xác"
            )
            event_type = "completed" if requested_id else "needs_info"
        else:
            result = format_tracking_result(order)
            event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result, attempt=attempt, call=call_number)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest_result = state.get("tool_results", [])[-1] if state.get("tool_results") else ""
    heuristic_result = _heuristic_tool_evaluation(latest_result)
    evaluation_result = heuristic_result or "needs_retry"
    reasoning = "explicit tool outcome resolved deterministically"
    latency_ms = 0
    llm_calls = 0
    structured_fallback = False
    evaluation_mode = "heuristic"

    should_use_judge = heuristic_result is None and os.getenv("LLM_EVALUATE", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    if should_use_judge:
        prompt = evaluation_prompt(latest_result)
        started = perf_counter()
        try:
            judged, llm_calls, structured_fallback = _invoke_structured(EvaluationResult, prompt)
            latency_ms = int((perf_counter() - started) * 1000)
            evaluation_result = judged.evaluation_result
            reasoning = judged.reasoning
            evaluation_mode = "llm_judge"
        except Exception as exc:  # LLM judge is optional; routing must remain reliable.
            latency_ms = int((perf_counter() - started) * 1000)
            reasoning = f"LLM judge fallback ({type(exc).__name__})"
            evaluation_mode = "safe_fallback"

    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"Kết quả công cụ được đánh giá là {evaluation_result}",
                latency_ms=latency_ms,
                reasoning=reasoning,
                evaluation_mode=evaluation_mode,
                llm_calls=llm_calls,
                prompt_version=PROMPT_VERSION,
                structured_fallback=structured_fallback,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    tool_context = "\n".join(state.get("tool_results", [])) or "No tool was required."
    approval = state.get("approval")
    approval_context = str(approval) if approval else "No approval decision was required."
    prompt = answer_prompt(
        query=state.get("query", ""),
        route=state.get("route", ""),
        tool_context=tool_context,
        approval_context=approval_context,
    )
    started = perf_counter()
    response, llm_calls = invoke_with_retry(get_llm(temperature=0), prompt)
    latency_ms = int((perf_counter() - started) * 1000)
    answer = _message_text(response)
    if not answer:
        raise RuntimeError("LLM returned an empty support answer")
    return {
        "final_answer": answer,
        "messages": [f"assistant:{answer}"],
        "events": [
            make_event(
                "answer",
                "completed",
                "Đã tạo câu trả lời bám sát dữ liệu",
                latency_ms=latency_ms,
                llm_calls=llm_calls,
                prompt_version=PROMPT_VERSION,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    approval = state.get("approval")
    if approval and not approval.get("approved", False):
        question = "Hành động không được duyệt. Tôi nên thực hiện phương án nào an toàn hơn?"
    else:
        question = (
            "Mình giúp được nhé. Bạn cho mình biết cụ thể vấn đề đang gặp và thông tin liên quan "
            "(ví dụ mã đơn hàng hoặc email tài khoản) được không?"
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "clarification_received": False,
        "messages": [f"assistant:{question}"],
        "events": [make_event("clarify", "completed", "Đã yêu cầu bổ sung thông tin")],
    }


def wait_for_user_node(state: AgentState) -> dict:
    """Pause an interactive chat and merge the user's reply into the same request.

    Batch evaluation remains deterministic: when interactive clarification is disabled,
    the graph simply finalizes with the clarification question as before.
    """
    if os.getenv("LANGGRAPH_CLARIFY_INTERRUPT", "false").lower() not in {"1", "true", "yes"}:
        return {
            "clarification_received": False,
            "events": [
                make_event(
                    "wait_for_user",
                    "skipped",
                    "Chế độ batch ghi nhận câu hỏi nhưng không chờ phản hồi trực tiếp",
                )
            ],
        }

    from langgraph.types import interrupt

    resumed = interrupt(
        {
            "kind": "clarification",
            "question": state.get("pending_question", "Bạn có thể nói rõ hơn không?"),
        }
    )
    if isinstance(resumed, dict):
        reply = str(resumed.get("answer", "")).strip()
    else:
        reply = str(resumed).strip()
    if not reply:
        return {
            "clarification_received": False,
            "events": [make_event("wait_for_user", "empty", "Chưa nhận được thông tin bổ sung")],
        }

    original_query = state.get("query", "").strip()
    combined_query = f"{original_query}\nThông tin người dùng bổ sung: {reply}"
    return {
        "query": combined_query,
        "route": "",
        "pending_question": None,
        "final_answer": None,
        "clarification_received": True,
        "messages": [f"user:{reply}"],
        "events": [
            make_event(
                "wait_for_user",
                "resumed",
                "Đã nhận thông tin bổ sung và chuyển lại cho bộ định tuyến",
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    proposed_action = (
        f"Thực hiện hành động có tác động thật sau khi được duyệt: {state.get('query', '')}"
    )
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "approval_required",
                "Đã chuẩn bị hành động rủi ro để kiểm duyệt",
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return an approval decision plus an append-only audit event.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() in {"1", "true", "yes"}:
        from langgraph.types import interrupt

        resumed = interrupt(
            {
                "question": "Phê duyệt hành động hỗ trợ được đề xuất này?",
                "proposed_action": state.get("proposed_action", ""),
            }
        )
        if isinstance(resumed, dict):
            decision = ApprovalDecision.model_validate(resumed)
        else:
            decision = ApprovalDecision(approved=bool(resumed), reviewer="human")
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Tự động phê duyệt để chạy lab theo cách xác định.",
        )
    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                "approved" if decision.approved else "rejected",
                decision.comment or "approval decision recorded",
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0) + 1
    error = f"Đã ghi nhận lỗi tạm thời; lên lịch thử lại lần {attempt}"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "scheduled", error, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    answer = (
        f"Không thể hoàn thành yêu cầu sau {attempt} lần thử. "
        "Yêu cầu đã được chuyển sang hỗ trợ để kiểm tra thủ công."
    )
    return {
        "final_answer": answer,
        "errors": ["Đã hết số lần retry; yêu cầu được chuyển sang dead letter."],
        "events": [make_event("dead_letter", "failed", "Đã hết số lần retry")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "events": [make_event("finalize", "completed", "Workflow đã hoàn tất")],
    }
