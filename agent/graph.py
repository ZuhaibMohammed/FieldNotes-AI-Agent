"""LangGraph workflow for bounded parallel research and report writing."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.nodes import comparer, planner, researcher, reflector, start_gap_research, writer, route_after_reflection, route_after_research
from agent.telemetry import instrument_node


class ResearchState(TypedDict, total=False):
    """Shared state for research, UI progress, and bounded execution."""
    topic: str
    user_request: str
    report_instructions: list[str]
    sub_questions: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    search_cache: dict[str, list[dict[str, Any]]]
    page_cache: dict[str, str]
    queries_tried: list[str]
    gaps: list[dict[str, Any]]
    unresolved_gaps: list[dict[str, Any]]
    verified_findings: list[dict[str, Any]]
    iteration: int
    max_iterations: int
    report: str
    model: str
    helper_model: str
    mode: str
    time_budget: float
    results_per_query: int
    queries_per_question: int
    pages_per_question: int
    time_expired: bool
    reflection_sufficient: bool
    comparison: dict[str, Any]
    current_step: str
    quick_claims: list[dict[str, Any]]
    quick_summaries: list[dict[str, Any]]
    quick_findings: dict[str, Any]
    quick_comparison: dict[str, Any]
    quick_limitations: dict[str, Any]
    quick_findings_failed: str
    quick_comparison_failed: str
    quick_limitations_failed: str
    quick_extract_warnings: list[str]


def build_graph(mode: str = "Deep") -> Any:
    """Build Quick or Deep graph; Quick mode is an isolated parallel pipeline."""
    if mode == "Quick":
        from agent.quick import assemble, extract, plan, research, sync_node, write_comparison, write_findings, write_limitations
        quick = StateGraph(ResearchState)
        for name, node in (("quick_plan", sync_node(plan)), ("quick_research", sync_node(research)), ("quick_extract", sync_node(extract)),
                           ("quick_write_findings", sync_node(write_findings)), ("quick_write_comparison", sync_node(write_comparison)),
                           ("quick_write_limitations", sync_node(write_limitations)), ("quick_assemble", assemble)):
            quick.add_node(name, instrument_node(name, node))
        quick.add_edge(START, "quick_plan")
        quick.add_edge("quick_plan", "quick_research")
        quick.add_edge("quick_research", "quick_extract")
        quick.add_edge("quick_extract", "quick_write_findings")
        quick.add_edge("quick_extract", "quick_write_comparison")
        quick.add_edge("quick_extract", "quick_write_limitations")
        quick.add_edge(["quick_write_findings", "quick_write_comparison", "quick_write_limitations"], "quick_assemble")
        quick.add_edge("quick_assemble", END)
        return quick.compile()
    builder = StateGraph(ResearchState)
    for name, node in (("planner", planner), ("researcher", researcher), ("reflector", reflector),
                       ("more_research", start_gap_research), ("comparer", comparer), ("writer", writer)):
        builder.add_node(name, instrument_node(name, node))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_conditional_edges("researcher", route_after_research, {"reflector":"reflector", "comparer":"comparer", "writer":"writer"})
    builder.add_conditional_edges("reflector", route_after_reflection, {"more_research":"more_research", "comparer":"comparer", "writer":"writer"})
    builder.add_edge("more_research", "researcher")
    builder.add_edge("comparer", "writer")
    builder.add_edge("writer", END)
    return builder.compile()
