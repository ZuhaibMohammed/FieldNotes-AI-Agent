"""Regression tests for bounded execution and page reads."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.graph import route_after_reflection
from agent import nodes
from agent.telemetry import RunTelemetry, activate_telemetry, reset_telemetry
from agent.tools import fetch_page


class _HangingHandler(BaseHTTPRequestHandler):
    """Delay local HTTP responses to exercise the hard page timeout."""

    def do_GET(self) -> None:  # noqa: N802
        time.sleep(1.2)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<html><body>late response</body></html>")

    def log_message(self, *_: object) -> None:
        """Silence test server logs."""


def test_hanging_fetch_returns_within_timeout() -> None:
    """A page endpoint that hangs cannot block beyond the configured timeout."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HangingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    started = time.monotonic()
    try:
        assert fetch_page(f"http://127.0.0.1:{server.server_port}/", timeout=0.2) == ""
        assert time.monotonic() - started < 0.8
    finally:
        server.shutdown()
        server.server_close()


def test_global_time_budget_routes_to_writer() -> None:
    """An expired global clock ends reflection and advances to the report."""
    telemetry = RunTelemetry("Quick", 0.01)
    token = activate_telemetry(telemetry)
    try:
        time.sleep(0.02)
        assert route_after_reflection({"reflection_sufficient":False, "gaps":[{"question":"Unresolved question"}], "iteration":0, "max_iterations":2}) == "writer"
    finally:
        reset_telemetry(token)


def test_source_comparison_is_last_report_section_before_source_links(monkeypatch: object) -> None:
    """The table stays before end matter and prose has no numeric source markers."""
    def fake_structured(*_: object, **__: object) -> dict[str, object]:
        return {"title":"Example", "executive_summary":"A concise summary.", "agreements_disagreements":"Evidence aligns.", "limitations":"One source only.", "verified_findings":[{"question":"Main question?", "claim":"Supported statement [1]", "status":"supported", "supporting_source_ids":[1], "rationale":""}]}
    monkeypatch.setattr(nodes, "_structured", fake_structured)
    state = {"topic":"Example", "model":"openai/gpt-oss-120b", "findings":[{"question":"Main question?", "summary":"A summary.", "source_ids":[1]}], "sources":[{"id":1, "title":"Example source", "url":"https://example.com", "domain":"example.com", "type":"primary", "snippet":"Support."}], "comparison":{"sources":[{"id":1,"name":"Example source","domain":"example.com","type":"primary","stance":"Supports the statement.","strengths":[],"caveats":[]}],"agreements":[],"disagreements":[]}}
    report = nodes.writer(state)["report"]
    body, source_links = report.split("## Sources", maxsplit=1)
    assert body.index("## Source Comparison") > body.index("## Open Questions/Limitations")
    assert not __import__("re").search(r"(?<!\w)\[\d+\]", body)
    assert "[1] [Example source]" in source_links


def test_deep_planner_falls_back_after_invalid_json(monkeypatch: object) -> None:
    """A malformed planner response still yields the requested Deep plan size."""
    def fail_structured(*_: object, **__: object) -> dict[str, object]:
        raise ValueError("Could not parse structured LLM response: No JSON object found")
    monkeypatch.setattr(nodes, "_structured", fail_structured)
    result = nodes.planner({"topic":"RAG vs fine-tuning for domain-specific QA", "mode":"Deep", "deep_question_count":4})
    assert len(result["sub_questions"]) == 4
    assert all(len(item["queries"]) == 2 for item in result["sub_questions"])
    assert "fallback plan" in result["current_step"]
