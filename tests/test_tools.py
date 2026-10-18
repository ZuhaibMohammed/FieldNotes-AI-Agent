"""Tests for search result normalization."""

from agent.tools import normalize_results


def test_normalize_results_deduplicates_and_maps_provider_fields() -> None:
    """Map href/body results and keep one valid entry per URL."""
    results = [
        {"title": "One", "href": "https://example.com/a", "body": "A snippet"},
        {"title": "Duplicate", "url": "https://example.com/a", "snippet": "ignored"},
        {"title": "Bad", "href": "javascript:void(0)"},
        {"title": "Two", "url": "https://example.org", "snippet": "B"},
    ]
    assert normalize_results(results, 5) == [
        {"title": "One", "url": "https://example.com/a", "snippet": "A snippet"},
        {"title": "Two", "url": "https://example.org", "snippet": "B"},
    ]

