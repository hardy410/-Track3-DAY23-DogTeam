"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .state import AgentState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Build and compile the LangGraph workflow.

    The graph implements this architecture:

    START → intake → classify → [conditional: route_after_classify]
      simple       → answer → finalize → END
      tool         → tool → evaluate → [conditional: route_after_evaluate]
                                          success → answer → finalize → END
                                          needs_retry → retry → [conditional: route_after_retry]
                                                                  tool (retry)
                                                                  dead_letter → finalize → END
      missing_info → clarify → finalize → END
      risky        → risky_action → approval → [conditional: route_after_approval]
                                                  approved → tool → evaluate → ...
                                                  rejected → clarify → finalize → END
      error        → retry → [conditional: route_after_retry] → ...

    Steps:
    1. Import StateGraph, START, END from langgraph.graph
    2. Create StateGraph(AgentState)
    3. Import and add all nodes from nodes.py (11 nodes total)
    4. Import and use routing functions from routing.py for conditional edges
    5. Add fixed edges (e.g., START→intake, intake→classify, tool→evaluate, etc.)
    6. Add conditional edges using add_conditional_edges()
    7. Compile with checkpointer: graph.compile(checkpointer=checkpointer)

    Reference: https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/
    """
    from langgraph.graph import END, START, StateGraph

    from .nodes import (
        answer_node,
        approval_node,
        ask_clarification_node,
        classify_node,
        dead_letter_node,
        evaluate_node,
        finalize_node,
        intake_node,
        retry_or_fallback_node,
        risky_action_node,
        tool_node,
    )
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
    )

    builder = StateGraph(AgentState)
    builder.add_node("intake", intake_node)
    builder.add_node("classify", classify_node)
    builder.add_node("tool", tool_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("answer", answer_node)
    builder.add_node("clarify", ask_clarification_node)
    builder.add_node("risky_action", risky_action_node)
    builder.add_node("approval", approval_node)
    builder.add_node("retry", retry_or_fallback_node)
    builder.add_node("dead_letter", dead_letter_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )
    builder.add_edge("tool", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"answer": "answer", "retry": "retry"},
    )
    builder.add_edge("risky_action", "approval")
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"tool": "tool", "clarify": "clarify"},
    )
    builder.add_conditional_edges(
        "retry",
        route_after_retry,
        {"tool": "tool", "dead_letter": "dead_letter"},
    )
    builder.add_edge("answer", "finalize")
    builder.add_edge("clarify", "finalize")
    builder.add_edge("dead_letter", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
