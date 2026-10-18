"""Tests for query planning, relevance filtering, reflection and verification."""

import json
from pathlib import Path

from agent.nodes import (
    filter_novel_reflection_gaps,
    filter_relevance_results,
    filter_verified_claims,
    normalize_plan_output,
    split_topic_and_instructions,
)

FIXTURE = Path(__file__).parent / "fixtures" / "remote_work_question.json"


def test_planner_separates_meta_instructions_and_sanitizes_queries() -> None:
    """Report directions stay out of all generated search queries."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    request = fixture["question"]
    topic, instructions = split_topic_and_instructions(request)
    assert "remote work" in topic.lower()
    assert instructions
    plan = normalize_plan_output(
        {
            "topic": request,
            "report_instructions": [],
            "sub_questions": [
                {"question": f"Remote work outcome {index} since 2020", "queries": [f"remote work outcome {index} 2020 2025 survey", f"remote work outcome {index} 2020 2025 NBER"]}
                for index in range(1, 4)
            ],
        },
        request,
    )
    queries = [query.lower() for item in plan["sub_questions"] for query in item["queries"]]
    assert plan["topic"] == topic
    assert plan["report_instructions"]
    assert all(not any(phrase in query for phrase in fixture["forbidden_query_phrases"]) for query in queries)


def test_relevance_filter_drops_results_below_two() -> None:
    """Irrelevant search results are excluded and their reason is retained."""
    results = [{"title": "Remote work study", "url": "https://a.test"}, {"title": "Study design guide", "url": "https://b.test"}]
    grades = [{"url": "https://a.test", "score": 3, "reason": "Directly addresses productivity."}, {"url": "https://b.test", "score": 1, "reason": "Discusses methodology only."}]
    kept, dropped = filter_relevance_results(results, grades)
    assert [item["url"] for item in kept] == ["https://a.test"]
    assert dropped[0]["relevance_reason"] == "Discusses methodology only."


def test_reflector_rejects_duplicate_queries() -> None:
    """Reflection gaps with no new search terms are rejected as unresolved."""
    accepted, rejected = filter_novel_reflection_gaps(
        [{"question": "How does remote work affect output?", "reason": "Find firm data.", "new_queries": ["remote work productivity firm data"]}],
        ["remote work productivity firm data"],
    )
    assert accepted == []
    assert any(item.get("unresolved") for item in rejected)


def test_faithfulness_filter_removes_unsupported_claim() -> None:
    """Only claims with valid supporting source ids survive verification."""
    claims = [
        {"question": "Productivity?", "claim": "A measured result.", "status": "supported", "supporting_source_ids": [1]},
        {"question": "Productivity?", "claim": "An invented result.", "status": "unsupported", "supporting_source_ids": [1]},
    ]
    supported, rejected = filter_verified_claims(claims, {"Productivity?": [1]}, {1})
    assert [item["claim"] for item in supported] == ["A measured result."]
    assert [item["claim"] for item in rejected] == ["An invented result."]
