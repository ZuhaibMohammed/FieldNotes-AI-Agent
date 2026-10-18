"""Quick-mode output, budget, and rate-tracker regressions."""

from __future__ import annotations

import re
import time
import json

from agent import quick
from agent.graph import build_graph
from agent.llm import _HEADER_RATE_TRACKER, _HeaderRateTracker
from agent.telemetry import RunTelemetry, activate_telemetry, reset_telemetry


def _stub_json(system: str, prompt: str, model: str, tokens: int, schema: type) -> dict:
    """Return valid fixture responses for each Quick-mode structured step."""
    if schema is quick.Plan:
        return {"sub_questions":["RAG mechanisms for domain QA", "Fine tuning results for domain QA", "RAG versus tuning evaluations"], "queries":["RAG domain QA primary research", "RAG domain QA official documentation", "fine tuning domain QA empirical study", "fine tuning domain QA practitioner guide", "RAG fine tuning benchmark research", "RAG fine tuning engineering guide"]}
    if schema is quick.Extraction:
        ids = [int(x) for x in re.findall(r'"id":\s*(\d+)', prompt)]
        sid = ids[0] if ids else 1
        return {"summaries":[{"source_id":sid,"summary":f"The source describes a concrete evaluation of domain question answering. It reports measured retrieval or adaptation results. The text discusses implementation choices and the evaluated task. The reported findings apply to the tested setting.","source_type":"research","stance":"Reports task-specific results."}], "claims":[{"text":"The source reports measured outcomes for domain question answering in its tested setting.","source_id":sid,"scope":f"study {sid} on domain QA"},{"text":"The source describes implementation choices relevant to retrieval or model adaptation.","source_id":sid,"scope":f"study {sid} on domain QA"}]}
    if schema is quick.Findings:
        available = sorted({int(x) for x in re.findall(r'"source_id":\s*(\d+)', prompt)}) or [1]
        return {"executive_summary":"The extracted sources describe domain QA methods and evaluations [1].", "findings":[{"text":f"Finding {i} summarizes a supplied claim about domain QA methods", "confidence":"Medium", "source_ids":[available[(i-1)%len(available)]]} for i in range(1,6)]}
    if schema is quick.Comparison or schema.__name__ == "ExactQuickComparison":
        evidence = json.loads(prompt.split("Evidence JSON: ",1)[1].split("\nReturn",1)[0])
        srcs = evidence["sources"]
        ids = [item["id"] for item in srcs]
        return {"sources":[{"id":item["id"],"name":item["title"],"domain":item["domain"],"type":"research","stance":"Reports evaluated QA results","strengths":["Includes task-specific results"],"caveats":["Scope is limited to tested setting"]} for item in srcs], "agreements":[{"text":"Sources report that performance depends on the evaluation setting.","source_ids":ids}], "disagreements":[{"text":"The studies evaluate different approaches and data conditions.","source_ids":ids}]}
    if schema is quick.Limitations:
        return {"limitations":["The sources cover different task and data conditions [1][2][3]."],"open_questions":["How costs compare under a shared workload remains to be evaluated [1][2]."],"not_established":["A universally superior approach is not established by these sources [1][2][3]."]}
    raise AssertionError(schema)


def _fixture_sources():
    """Build three distinct real-text source fixtures."""
    return [{"id":i,"question":q,"title":f"Research paper {i}","url":f"https://journal{i}.org/study","domain":f"journal{i}.org","snippet":f"Measured domain question-answering evaluation number {i}.","text":f"This research paper reports measured domain question-answering results in study {i}. The evaluation describes a retrieval system and adaptation baseline.","excerpt":f"This research paper reports measured domain question-answering results in study {i}. The evaluation describes a retrieval system and adaptation baseline.","type":"other","relevance_score":2.0} for i,q in enumerate(["RAG mechanisms for domain QA", "Fine tuning results for domain QA", "RAG versus tuning evaluations"],1)]


def _patch_pipeline(monkeypatch, slow: bool = False) -> list[str]:
    """Patch external calls, returning the writer nodes observed in execution."""
    questions = ["RAG mechanisms for domain QA", "Fine tuning results for domain QA", "RAG versus tuning evaluations"]
    monkeypatch.setattr(quick, "_call_json", _stub_json)
    def fake_search(query: str, max_results: int):
        if slow and "fine tuning" in query:
            time.sleep(1.2)
            return []
        i = 3 if "primary" in query or "benchmark" in query else 1 if "RAG domain" in query else 2
        return [{"title":f"Research paper {i}","url":f"https://journal{i}.org/study","snippet":f"Measured domain QA results {i}."}]
    monkeypatch.setattr(quick, "search", fake_search)
    async def fake_fetch(urls, timeout=5.0, max_chars=3000):
        return {url:"Primary text describes a concrete evaluation and measured outcomes for domain question answering in the tested setting. Retrieval and adaptation conditions are discussed." for url in urls}
    monkeypatch.setattr(quick, "fetch_pages", fake_fetch)
    return []


def _state(budget: int = 90) -> dict:
    """Initial Quick graph state."""
    return {"topic":"RAG vs fine-tuning for domain-specific question answering", "sources":[], "findings":[], "gaps":[], "iteration":0,"max_iterations":0,"model":"openai/gpt-oss-120b","helper_model":"openai/gpt-oss-20b","mode":"Quick","time_budget":budget,"results_per_query":3,"queries_per_question":1,"pages_per_question":2}


def test_quick_mock_report_has_cited_findings_and_nonempty_comparison(monkeypatch) -> None:
    """Mocked evidence produces cited findings, populated comparison, and comparisons at the end."""
    _patch_pipeline(monkeypatch)
    # Stable single-source-per-question research fixtures.
    monkeypatch.setattr(quick, "search", lambda query, max_results: [{"title":"Empirical domain QA evaluation", "url":f"https://journal{3 if 'primary' in query or 'benchmark' in query else 1 if 'RAG domain' in query else 2}.org/study", "snippet":"Measured domain question answering evaluation."}])
    telemetry = RunTelemetry("Quick", 90)
    token = activate_telemetry(telemetry)
    try:
        final = build_graph("Quick").invoke(_state())
    finally:
        reset_telemetry(token)
    report = final["report"]
    body, _ = report.split("## Sources", 1)
    findings = body.split("## Key Findings",1)[1].split("## Agreements vs Disagreements",1)[0]
    assert len(re.findall(r"^- .*\[\d+\]", findings, re.M)) == 5
    assert "| Source |" in body and "Empirical domain QA evaluation" in body
    assert "**Agreements**" in body and "**Disagreements**" in body
    assert "No evidence available" not in report and "Absence of data limits conclusions" not in report
    assert body.index("## Source Comparison") > body.index("## Open Questions/Limitations")


def test_slow_search_does_not_starve_extract_or_write(monkeypatch) -> None:
    """Search cutoff is soft for later extract and write phases."""
    _patch_pipeline(monkeypatch, slow=True)
    telemetry = RunTelemetry("Quick", 2)
    token = activate_telemetry(telemetry)
    started = time.monotonic()
    try:
        state = _state(2)
        state["search_grace_seconds"] = 0.2
        final = build_graph("Quick").invoke(state)
    finally:
        reset_telemetry(token)
    assert final.get("report")
    assert any(x["Node"] == "quick_extract" for x in telemetry.node_table())
    assert any(x["Node"] == "quick_write_findings" for x in telemetry.node_table())
    assert time.monotonic()-started < 2  # slow search doesn't hold up extraction and writing


def test_unknown_header_tracker_does_not_proactively_wait() -> None:
    """Without actual provider headers, Quick pacing is immediate."""
    model = "test-model-without-headers"
    started = time.monotonic()
    assert _HEADER_RATE_TRACKER.wait_if_needed(model, 5000) == 0
    assert time.monotonic()-started < 2


def test_rate_tracker_uses_per_model_headers_and_paces_only_when_needed(monkeypatch) -> None:
    """Observed provider headers pace only the matching model's next oversized call."""
    import agent.llm as llm_module
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    tracker = _HeaderRateTracker()
    tracker.update("model-a", {"x-ratelimit-remaining-tokens":"10", "x-ratelimit-limit-tokens":"100", "x-ratelimit-reset-tokens":"5s"})
    assert tracker.wait_if_needed("model-a", 20, max_wait=0.25) == 0.25
    assert tracker.wait_if_needed("model-b", 20, max_wait=0.25) == 0
    assert sleeps == [0.25]


def test_daily_quota_failure_blocks_that_model_for_later_quick_calls() -> None:
    """An observed provider daily-quota reset prevents repeated doomed requests."""
    tracker = _HeaderRateTracker()
    tracker.mark_unavailable("model-daily-limited", 60)
    assert tracker.is_unavailable("model-daily-limited")
    assert not tracker.is_unavailable("different-model")


def test_telemetry_counts_actual_model_attempts() -> None:
    """Skipped daily-quota models do not inflate the real request count."""
    telemetry = RunTelemetry("Quick", 90)
    telemetry.record({"kind":"llm", "attempt_count":1, "fallback_used":True})
    assert telemetry.counts()["llm_calls"] == 1


def test_quick_extract_schema_requires_claims() -> None:
    """Quick extraction schema rejects empty claim results."""
    import pytest
    with pytest.raises(Exception):
        quick.Extraction.model_validate({"summaries":[],"claims":[]})


def test_quick_model_json_shape_variants_are_coerced() -> None:
    """List summaries and point-labeled comparison JSON validate without another call."""
    extraction = quick._coerce_schema_payload({"summaries":[{"source_id":1,"summary":["RAG retrieves external documents for context.","Fine-tuning updates model parameters using domain data.","The article contrasts these adaptation strategies."],"source_type":"article","stance":"Compares both methods"}],"claims":[{"claim":"RAG retrieves external documents for context.","source_id":1,"scope":"one QA benchmark"},{"claim":"Fine-tuning updates model parameters using domain data.","source_id":1,"scope":"one QA benchmark"}]},quick.Extraction)
    assert quick.Extraction.model_validate(extraction)
    exact = quick.create_model("ExactQuickComparison",__base__=quick.Comparison,sources=(list[quick.ComparisonRow],quick.Field(min_length=1,max_length=1)))
    comparison = quick._coerce_schema_payload({"sources":[{"id":1,"stance":"Retrieval changes context","strengths":"Provides cited context","caveats":"Depends on indexed sources"}],"agreements":[{"point":"Sources describe both retrieval and weight adaptation.","source_ids":[1]}],"disagreements":[]},exact)
    assert exact.model_validate(comparison)


def test_limitations_remove_contradicted_no_evidence_claim() -> None:
    """A broad no-evidence sentence is rewritten when extracted claims cover the topic."""
    state = {"quick_claims":[{"text":"Fine tuning performance characteristics of language models depend on data volume and compute resources.","source_id":1}], "sources":[{"id":1}], "quick_limitations":{"limitations":["No evidence provided on fine-tuning performance characteristics of language models."],"open_questions":[],"not_established":[]}}
    result = quick._sanitize_limitations(state)
    assert all("No evidence provided" not in item for item in result["limitations"])


def test_summary_normalizes_parenthetical_and_missing_citations() -> None:
    """Every executive-summary sentence receives valid bracket citations."""
    state = {"quick_claims":[{"text":"Retrieval augmented generation retrieves current information for domain question answering.","source_id":1},{"text":"Fine tuning adapts model weights for specialized terminology.","source_id":2}],"sources":[{"id":1},{"id":2}]}
    result = quick._cite_summary("RAG can retrieve current information (1,2). Fine tuning changes model weights.", state)
    assert "(1,2)" not in result
    assert re.search(r"current information \[1\]\[2\]", result)
    assert re.search(r"weights\. \[2\]", result)
    assert "Evidence [1][2]." == quick._normalize_citation_groups("Evidence [1,2].", {1,2})
    grouped = quick._normalize_parenthetical_citations("Evidence ([1](<https://one.test>),2).", {1,2})
    assert grouped == "Evidence [1][2]."


def test_writer_failure_keeps_claim_backed_report_and_structured_fallbacks(monkeypatch) -> None:
    """Failed report calls retain diagnostics while code assembles evidence-backed sections."""
    import asyncio
    from agent.quick import QuickStepFailure
    def fail(*args, **kwargs):
        raise QuickStepFailure("malformed JSON")
    monkeypatch.setattr(quick, "_call_json", fail)
    sources = _fixture_sources()
    claims = [{"text":f"Extracted claim {i} reports concrete domain QA results in the tested setting.","source_id":i%3+1} for i in range(7)]
    summaries = [{"source_id":s["id"],"summary":"This source reports a concrete domain QA evaluation with measured outcomes and a bounded test setting.","source_type":"research","stance":"Reports a measured domain QA result."} for s in sources]
    state = {"topic":"RAG compared with fine-tuning for domain QA", "model":"openai/gpt-oss-120b", "sources":sources, "quick_claims":claims, "quick_summaries":summaries}
    for node in (quick.write_findings, quick.write_comparison, quick.write_limitations):
        state.update(asyncio.run(node(state)))
    report = quick.assemble(state)["report"]
    assert "## Source Comparison" in report and "| Source |" in report
    assert "[1](<https://journal1.org/study>)" in report
    assert "The extracted claims describe evidence" in report
    assert "The retrieved claims do not establish a universal winner" in report
    assert "failed after one retry" not in report
    assert state.get("quick_comparison_failed") and state.get("quick_limitations_failed")
    assert "No evidence available" not in report
