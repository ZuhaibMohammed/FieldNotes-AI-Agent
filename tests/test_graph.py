"""Tests for LangGraph routing decisions."""

from agent.graph import route_after_reflection, route_after_research


def test_parallel_research_routes_to_reflection_or_writer() -> None:
    """A complete parallel batch reflects unless the time budget has expired."""
    assert route_after_research({"sub_questions": ["a", "b"]}) == "reflector"
    assert route_after_research({"time_expired": True}) == "writer"


def test_reflection_routes_to_research_or_comparison() -> None:
    """Gaps route to another round only within the iteration limit."""
    assert route_after_reflection({"reflection_sufficient": False, "gaps": ["gap"], "iteration": 0, "max_iterations": 2}) == "more_research"
    assert route_after_reflection({"reflection_sufficient": False, "gaps": ["gap"], "iteration": 2, "max_iterations": 2}) == "comparer"
    assert route_after_reflection({"reflection_sufficient": True, "gaps": [], "iteration": 0, "max_iterations": 2}) == "comparer"
