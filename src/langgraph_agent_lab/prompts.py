"""Versioned prompt templates for the support workflow."""

from __future__ import annotations

PROMPT_VERSION = "2026-08-25.v1"


def classification_prompt(query: str) -> str:
    """Build the intent-classification prompt."""
    return f"""You route customer-support tickets into exactly one category.

Categories:
- risky: requests that cause side effects, including refunds, deletion, cancellation,
  changing data, or sending messages/emails.
- tool: information lookup, tracking, search, or retrieval without side effects.
- missing_info: vague or incomplete requests with insufficient actionable context.
- error: reports of timeouts, crashes, unavailable services, or system failures.
- simple: general support questions answerable without a tool or side effect.

When multiple categories apply, use this strict priority:
risky > tool > missing_info > error > simple.

Ticket: {query}
"""


def evaluation_prompt(tool_result: str) -> str:
    """Build the optional LLM-as-judge prompt."""
    return f"""Evaluate this support-tool result.
Return needs_retry only when it indicates an error, timeout, incomplete operation,
or unusable result. Otherwise return success.

Tool result: {tool_result}
"""


def answer_prompt(*, query: str, route: str, tool_context: str, approval_context: str) -> str:
    """Build the grounded support-answer prompt."""
    return f"""You are a concise customer-support assistant.
Answer the ticket using only the supplied context. Do not invent order details,
execution results, or policies. Clearly distinguish guidance from completed actions.

Ticket: {query}
Route: {route}
Tool context: {tool_context}
Approval context: {approval_context}
"""
