"""Research graph nodes, validation helpers, and report assembly."""

from __future__ import annotations

import json
import re
import asyncio
import math
import threading
from typing import Any, Literal
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator

from agent import prompts
from agent.llm import FALLBACK_MODEL, content_text, invoke_with_fallback, make_llm, normalize_citations, parse_json_response
from agent.tools import domain_name, fetch_page, fetch_pages, search
from agent.telemetry import current_telemetry


class PlannedQuestion(BaseModel):
    """One topical question and its two keyword search queries."""
    question: str = Field(min_length=8)
    queries: list[str] = Field(min_length=1, max_length=2)


class PlanOutput(BaseModel):
    """Structured research plan with report-only instructions separated."""
    topic: str = Field(min_length=3)
    report_instructions: list[str] = Field(default_factory=list, max_length=6)
    sub_questions: list[PlannedQuestion] = Field(min_length=3, max_length=5)

    @field_validator("report_instructions", mode="before")
    @classmethod
    def normalize_report_instructions(cls, value: Any) -> Any:
        """Accept a single instruction string as well as a JSON string list."""
        return [value] if isinstance(value, str) else value


class RelevanceGrade(BaseModel):
    """A result's relevance score and coarse source-quality label."""
    url: str
    score: int = Field(ge=0, le=3)
    reason: str = Field(min_length=3)
    source_type: Literal["primary", "review", "news", "blog", "reference", "other"] = "other"


class RelevanceOutput(BaseModel):
    """Batch relevance assessment for one sub-question."""
    grades: list[RelevanceGrade]


class QueryRewriteOutput(BaseModel):
    """One materially different search query."""
    query: str = Field(min_length=4)


class ReflectionGap(BaseModel):
    """A concrete gap and new queries for a reflection round."""
    question: str = Field(min_length=8)
    reason: str = Field(min_length=4)
    new_queries: list[str] = Field(min_length=1, max_length=3)


class ReflectionOutput(BaseModel):
    """Structured evidence sufficiency decision."""
    sufficient: bool
    gaps: list[ReflectionGap] = Field(default_factory=list, max_length=3)


class ComparisonSource(BaseModel):
    """Comparison and quality notes for one retrieved source."""
    id: int
    name: str
    domain: str
    type: str
    stance: str
    strengths: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ComparisonPoint(BaseModel):
    """An agreement or disagreement tied to supporting source ids."""
    text: str
    source_ids: list[int] = Field(default_factory=list)


class ComparisonOutput(BaseModel):
    """Structured source comparison."""
    sources: list[ComparisonSource] = Field(default_factory=list)
    agreements: list[ComparisonPoint] = Field(default_factory=list)
    disagreements: list[ComparisonPoint] = Field(default_factory=list)


class VerifiedClaim(BaseModel):
    """Claim-level support assessment."""
    question: str
    claim: str
    status: Literal["supported", "partially supported", "unsupported"]
    supporting_source_ids: list[int] = Field(default_factory=list)
    rationale: str = ""


class FaithfulnessOutput(BaseModel):
    """Batch verification results for all draft claims."""
    claims: list[VerifiedClaim]


class ReportNarrative(BaseModel):
    """Report prose and faithfulness results returned in one model pass."""
    title: str
    executive_summary: str
    agreements_disagreements: str
    limitations: str
    verified_findings: list[VerifiedClaim] = Field(default_factory=list)


def _structured(
    schema: type[BaseModel],
    system: str,
    prompt: str,
    model_name: str | None = None,
    retries: int = 1,
) -> dict[str, Any]:
    """Request and validate JSON, retrying one parse/schema failure."""
    output_limits = {"PlanOutput":1400, "ReportNarrative":1400, "ComparisonOutput":1100, "ReflectionOutput":700, "RelevanceOutput":650, "QueryRewriteOutput":350}
    llm = make_llm(model_name, max_tokens=output_limits.get(schema.__name__, 650))
    error: Exception | None = None
    for attempt in range(retries + 1):
        suffix = " Return only valid JSON matching the requested schema." if attempt else ""
        try:
            response = invoke_with_fallback(
                llm,
                [SystemMessage(content=system), HumanMessage(content=prompt + suffix)],
                json_mode=True,
            )
            return schema.model_validate(parse_json_response(content_text(response))).model_dump()
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            error = exc
    raise ValueError(f"Could not parse structured LLM response: {error}")


META_QUERY_PATTERNS = (
    r"\bcross[\s-]?sectional\b",
    r"\blongitudinal\b",
    r"\btypes?\s+of\s+stud(?:y|ies)\b",
    r"\bstudy\s+types?\b",
    r"\bcompare findings across\b[^,.;]*",
    r"\bnote where (?:the )?evidence is mixed\b",
)


def split_topic_and_instructions(request: str) -> tuple[str, list[str]]:
    """Separate report directions from the searchable research topic."""
    text = " ".join(request.split())
    match = re.search(
        r"\b(compare findings across|compare across different types|note where (?:the )?evidence is mixed|"
        r"compare findings and note where)\b",
        text,
        flags=re.I,
    )
    if not match:
        return text.strip(" ."), []
    topic = text[: match.start()].rstrip(" ,;.-")
    instructions = text[match.start() :].strip(" .")
    return topic or text, [instructions] if instructions else []


def sanitize_search_query(query: str) -> str:
    """Remove meta-report language that does not belong in a search query."""
    cleaned = query
    for pattern in META_QUERY_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:compare|discuss|note)\s+(?:the\s+)?(?:findings|mixed evidence)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"[\"'`]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def _keyword_query(text: str) -> str:
    """Create a short keyword query when a model query is unusable."""
    cleaned = sanitize_search_query(text)
    cleaned = re.sub(r"\b(?:how|what|when|where|why|does|do|the|a|an|and|or|of|to|for|with|since)\b", " ", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip(" ,;:-")


def normalize_plan_output(payload: dict[str, Any], user_request: str, query_count: int = 2, question_count: int | None = None) -> dict[str, Any]:
    """Normalize planner output to mode-specific questions and query counts."""
    request_topic, inferred_instructions = split_topic_and_instructions(user_request)
    plan_topic, plan_instructions = split_topic_and_instructions(str(payload.get("topic") or request_topic))
    topic = plan_topic or request_topic
    instructions = list(dict.fromkeys([*inferred_instructions, *payload.get("report_instructions", []), *plan_instructions]))
    normalized_questions: list[dict[str, Any]] = []
    for item in payload.get("sub_questions", []):
        question = str(item.get("question", "")).strip()
        raw_queries = item.get("queries", [])
        queries = list(dict.fromkeys(q for q in (sanitize_search_query(str(q)) for q in raw_queries) if q))
        base = _keyword_query(question or topic)
        if len(queries) < query_count:
            alternatives = [f"{base} outcomes survey data 2020 2025", f"{base} research findings evidence 2020 2025"]
            queries = list(dict.fromkeys([*queries, *(q for q in alternatives if q not in queries)]))[:query_count]
        normalized_questions.append({"question": question or topic, "queries": queries[:query_count]})
    if question_count:
        normalized_questions = normalized_questions[:question_count]
        stems = ["What evidence supports the main benefits", "What limitations or risks affect the outcomes", "What implementation and cost trade-offs matter", "Which use cases are best suited to each approach", "What changes when evidence is applied in practice"]
        while len(normalized_questions) < question_count:
            stem = stems[len(normalized_questions) % len(stems)]
            question = f"{stem} for {topic}?"
            query = _keyword_query(f"{topic} {stem} evidence")
            normalized_questions.append({"question":question, "queries":[query] if query_count == 1 else [query, f"{query} review comparison"]})
    return {"topic": topic, "report_instructions": instructions, "sub_questions": normalized_questions}


def _fallback_plan(user_request: str, topic: str, question_count: int, query_count: int) -> dict[str, Any]:
    """Build a usable, bounded plan when the planner response is not valid JSON."""
    dimensions = [
        ("core approaches and how they work", "overview mechanisms"),
        ("effectiveness and the outcomes reported", "effectiveness outcomes evidence"),
        ("practical costs, limitations, and risks", "cost limitations risks"),
        ("data and implementation requirements", "data implementation requirements"),
        ("use cases and decision criteria", "use cases comparison"),
    ]
    questions: list[dict[str, Any]] = []
    for title, terms in dimensions[:question_count]:
        question = f"What does the evidence show about {terms.replace(' ', ' ')} for {topic}?"
        query = _keyword_query(f"{topic} {terms}")
        queries = [query]
        if query_count > 1:
            queries.append(_keyword_query(f"{topic} {terms} review comparison"))
        questions.append({"question":question, "queries":queries})
    return normalize_plan_output({"topic":topic, "report_instructions":[], "sub_questions":questions}, user_request, query_count, question_count)


def planner(state: dict[str, Any]) -> dict[str, Any]:
    """Plan topical questions and keyword queries, separating report guidance."""
    user_request = str(state.get("user_request") or state.get("topic", ""))
    topic_guess, instructions_guess = split_topic_and_instructions(user_request)
    mode = state.get("mode", "Deep")
    quick = mode == "Quick"
    question_count = 3 if quick else int(state.get("deep_question_count", 4))
    query_count = 1 if quick else 2
    prompt = json.dumps({"user_request": user_request, "topic_guess": topic_guess, "report_instructions_guess": instructions_guess, "mode":mode, "question_count":question_count, "queries_per_question":query_count}, ensure_ascii=False)
    try:
        payload = _structured(PlanOutput, prompts.PLANNER, prompt, model_name=state.get("model"), retries=1)
        plan = normalize_plan_output(payload, user_request, query_count, question_count)
        plan_note = "Planning complete"
    except ValueError as exc:
        plan = _fallback_plan(user_request, topic_guess, question_count, query_count)
        plan_note = "Used a compact fallback plan after the planner returned invalid structured output"
        telemetry = current_telemetry()
        if telemetry:
            telemetry.record({"kind":"structured_fallback", "name":"planner", "model":state.get("model", ""), "error":str(exc)[:250]})
    return {
        **plan,
        "research_index": 0,
        "query_retry_count": 0,
        "active_queries": [],
        "queries_tried": [],
        "current_step": plan_note,
        "search_cache": {},
        "page_cache": {},
        "time_expired": False,
    }


def _tokens(query: str) -> set[str]:
    """Return informative lowercase words for query similarity checks."""
    stop = {"the", "and", "for", "with", "from", "that", "this", "study", "research", "data", "findings"}
    return {word for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) > 2 and word not in stop}


def filter_relevance_results(results: list[dict[str, Any]], grades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split search results at relevance score 2 and retain filter reasons."""
    grade_by_url = {str(item.get("url", "")): item for item in grades}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for result in results:
        grade = grade_by_url.get(str(result.get("url", "")), {})
        enriched = {**result, "relevance_score": int(grade.get("score", 0)), "relevance_reason": str(grade.get("reason", "No relevance assessment was returned.")), "source_type": grade.get("source_type", "other")}
        (kept if enriched["relevance_score"] >= 2 else dropped).append(enriched)
    return kept, dropped


def _current_question(state: dict[str, Any]) -> dict[str, Any]:
    """Read the current sub-question in mapping form."""
    item = state.get("sub_questions", [])[state.get("research_index", 0)]
    return item if isinstance(item, dict) else {"question": str(item), "queries": []}


def researcher(state: dict[str, Any]) -> dict[str, Any]:
    """Search planned queries and collect deduplicated normalized results."""
    question = _current_question(state)
    queries = state.get("active_queries") or question.get("queries", [])
    queries_tried = list(state.get("queries_tried", []))
    query_log = list(state.get("query_log", []))
    raw_results: list[dict[str, Any]] = []
    seen: set[str] = set()
    count = int(state.get("results_per_query", 5))
    for raw_query in queries:
        query = sanitize_search_query(str(raw_query))
        if not query or is_duplicate_query(query, queries_tried):
            continue
        queries_tried.append(query)
        try:
            hits = search(query, max_results=count)
            error = ""
        except Exception as exc:
            hits = []
            error = str(exc)[:240]
        duplicates = 0
        for hit in hits:
            url = str(hit.get("url", ""))
            if url and url not in seen:
                seen.add(url)
                raw_results.append({**hit, "query": query})
            elif url:
                duplicates += 1
        query_log.append({"question": question["question"], "query": query, "returned": len(hits), "kept": 0, "filtered": 0, "duplicates": duplicates, "error": error})
    return {
        "raw_results": raw_results,
        "kept_results": [],
        "queries_tried": queries_tried,
        "query_log": query_log,
        "active_queries": list(queries),
        "current_step": f"Searched {len(queries)} queries for: {question['question']}",
    }


def _fallback_grade(question: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use conservative lexical overlap if batched grading is unavailable."""
    q_tokens = _tokens(question)
    grades = []
    for result in results:
        r_tokens = _tokens(f"{result.get('title', '')} {result.get('snippet', '')}")
        overlap = len(q_tokens & r_tokens)
        grades.append({"url": result.get("url", ""), "score": 2 if overlap >= 2 else 1 if overlap else 0,
                       "reason": "Fallback keyword overlap; automated relevance review was unavailable.", "source_type": "other"})
    return grades


def grade_relevance(state: dict[str, Any]) -> dict[str, Any]:
    """Batch-grade one sub-question's search results and retain audit details."""
    question = _current_question(state)["question"]
    raw_results = state.get("raw_results", [])
    grades: list[dict[str, Any]] = []
    if raw_results:
        payload = {"question": question, "results": [{"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("snippet", "")[:350]} for item in raw_results]}
        try:
            output = _structured(RelevanceOutput, prompts.RELEVANCE, json.dumps(payload, ensure_ascii=False), model_name=state.get("model"))
            grades = output["grades"]
        except Exception:
            grades = _fallback_grade(question, raw_results)
    kept, dropped = filter_relevance_results(raw_results, grades)
    all_dropped = [*state.get("dropped_results", []), *dropped]
    query_log = [dict(item) for item in state.get("query_log", [])]
    current_urls = {item.get("url") for item in raw_results}
    kept_urls = {item.get("url") for item in kept}
    for log in query_log:
        if log.get("question") == question:
            hits = [item for item in raw_results if item.get("query") == log.get("query")]
            log["kept"] = sum(item.get("url") in kept_urls for item in hits)
            log["filtered"] = sum(item.get("url") in current_urls and item.get("url") not in kept_urls for item in hits)
    needs_rewrite = len(kept) < 2 and state.get("query_retry_count", 0) == 0
    return {
        "kept_results": kept,
        "dropped_results": all_dropped,
        "query_log": query_log,
        "needs_rewrite": needs_rewrite,
        "relevance_count": len(kept),
        "filtered_count": len(dropped),
        "current_step": f"Filtered results: {len(kept)} kept, {len(dropped)} off-topic",
    }


def query_rewriter(state: dict[str, Any]) -> dict[str, Any]:
    """Generate one new query when relevance grading keeps fewer than two."""
    question = _current_question(state)["question"]
    tried = list(state.get("queries_tried", []))
    payload = {"topic": state.get("topic", ""), "question": question, "queries_tried": tried,
               "rejected_results": [{"title": item.get("title", ""), "reason": item.get("relevance_reason", "")} for item in state.get("dropped_results", [])[-8:]]}
    prompt = json.dumps(payload, ensure_ascii=False)
    candidate = ""
    for attempt in range(2):
        try:
            output = _structured(QueryRewriteOutput, prompts.QUERY_REWRITE, prompt, model_name=state.get("model"))
            candidate = sanitize_search_query(output["query"])
        except Exception:
            candidate = ""
        if candidate and not is_duplicate_query(candidate, tried):
            break
        tried.append(candidate or " ")
        prompt = json.dumps({**payload, "note": "The last proposed query duplicated earlier wording. Use different concrete topic keywords and a different evidence/source type."}, ensure_ascii=False)
        candidate = ""
    if not candidate:
        unresolved = [*state.get("unresolved_gaps", []), {"question": question, "reason": "No materially new search query could be generated after the relevance retry."}]
        return {"rewrite_success": False, "query_retry_count": 1, "unresolved_gaps": unresolved, "current_step": "No distinct query available"}
    return {
        "active_queries": [candidate], "query_retry_count": 1, "raw_results": [], "kept_results": [],
        "needs_rewrite": False, "rewrite_success": True, "current_step": f"Rewritten query: {candidate}",
    }


def reader(state: dict[str, Any]) -> dict[str, Any]:
    """Read only relevant pages, cache them per run, and summarize evidence."""
    question = _current_question(state)["question"]
    results = state.get("kept_results", [])[:3]
    sources = [dict(item) for item in state.get("sources", [])]
    source_by_url = {item["url"]: item for item in sources}
    page_cache = dict(state.get("page_cache", {}))
    evidence: list[dict[str, Any]] = []
    for result in results:
        url = result["url"]
        source = source_by_url.get(url)
        if source is None:
            page_text = page_cache[url] if url in page_cache else fetch_page(url)
            page_cache[url] = page_text
            source = {
                **result,
                "id": len(sources) + 1,
                "domain": domain_name(url),
                "text": page_text,
                "type": result.get("source_type", "other"),
                "relevance_score": result.get("relevance_score", 2),
                "relevance_reason": result.get("relevance_reason", ""),
            }
            sources.append(source)
            source_by_url[url] = source
        content = source.get("text") or result.get("snippet", "")
        if content:
            evidence.append({"id": source["id"], "title": source["title"], "type": source.get("type", "other"),
                             "score": source.get("relevance_score", 2), "kind": "page text" if source.get("text") else "search-result snippet only",
                             "text": content[:1500]})

    unresolved = [dict(item) for item in state.get("unresolved_gaps", [])]
    findings = list(state.get("findings", []))
    if not evidence:
        summary = "This sub-question remains unresolved after the planned queries and one relevance-focused rewrite returned no usable results."
        unresolved = [item for item in unresolved if item.get("question") != question]
        unresolved.append({"question": question, "reason": "No relevant results remained after one targeted query rewrite."})
    else:
        evidence_payload = "\n\n".join(
            f"[{item['id']}] {item['title']} | quality type: {item['type']} | relevance {item['score']}/3 | {item['kind']}\n{item['text']}"
            for item in evidence
        )
        llm = make_llm(state.get("model"))
        response = invoke_with_fallback(
            llm,
            [SystemMessage(content=prompts.SUMMARY), HumanMessage(content=f"Topic: {state.get('topic', '')}\nQuestion: {question}\n\nEvidence:\n{evidence_payload}")],
        )
        summary = content_text(response)
        unresolved = [item for item in unresolved if item.get("question") != question]
    finding = {
        "question": question,
        "summary": summary,
        "source_ids": [item["id"] for item in evidence],
        "search_queries": [item["query"] for item in state.get("query_log", []) if item.get("question") == question],
    }
    findings.append(finding)
    return {
        "findings": findings,
        "sources": sources,
        "page_cache": page_cache,
        "unresolved_gaps": unresolved,
        "research_index": state.get("research_index", 0) + 1,
        "query_retry_count": 0,
        "active_queries": [],
        "raw_results": [],
        "kept_results": [],
        "needs_rewrite": False,
        "current_step": f"Read {len(evidence)} relevant items for: {question}",
    }


def _query_terms(query: str) -> set[str]:
    """Token set for detecting substantially repeated query strings."""
    return _tokens(query)


def is_duplicate_query(query: str, earlier_queries: list[str], threshold: float = 0.72) -> bool:
    """Detect exact or near-duplicate query wording."""
    query_tokens = _query_terms(sanitize_search_query(query))
    if not query_tokens:
        return True
    for earlier in earlier_queries:
        previous = _query_terms(sanitize_search_query(earlier))
        if query_tokens == previous:
            return True
        union = query_tokens | previous
        if union and len(query_tokens & previous) / len(union) >= threshold:
            return True
    return False


def filter_novel_reflection_gaps(gaps: list[dict[str, Any]], queries_tried: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep reflection gaps with at least one meaningfully new query."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    known = list(queries_tried)
    for gap in gaps:
        novel: list[str] = []
        for raw_query in gap.get("new_queries", []):
            query = sanitize_search_query(str(raw_query))
            if query and not is_duplicate_query(query, known):
                novel.append(query)
                known.append(query)
            else:
                rejected.append({"question": gap.get("question", ""), "query": raw_query, "reason": "Duplicate or meta-instruction query rejected."})
        if novel:
            accepted.append({"question": gap["question"], "reason": gap.get("reason", ""), "new_queries": novel})
        elif gap.get("question"):
            rejected.append({"question": gap["question"], "reason": gap.get("reason", "No novel query remained."), "unresolved": True})
    return accepted, rejected


def reflector(state: dict[str, Any]) -> dict[str, Any]:
    """Find focused evidence gaps and reject previously tried queries."""
    payload = {
        "topic": state.get("topic", ""),
        "report_instructions": state.get("report_instructions", []),
        "findings": [{"question": item["question"], "summary": item["summary"][:700], "source_ids": item.get("source_ids", [])} for item in state.get("findings", [])],
        "queries_tried": state.get("queries_tried", []),
        "unresolved_gaps": state.get("unresolved_gaps", []),
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    try:
        output = _structured(ReflectionOutput, prompts.REFLECTOR, prompt, model_name=state.get("model"))
        accepted, rejected = filter_novel_reflection_gaps(output["gaps"], state.get("queries_tried", []))
        if rejected:
            retry_prompt = json.dumps({**payload, "rejected_duplicate_queries": rejected, "instruction": "Regenerate the gaps with genuinely different keywords, evidence types, or source types."}, ensure_ascii=False)
            try:
                retry = _structured(ReflectionOutput, prompts.REFLECTOR, retry_prompt, model_name=state.get("model"))
                accepted, rejected = filter_novel_reflection_gaps(retry["gaps"], state.get("queries_tried", []))
                output = retry
            except Exception:
                pass
    except Exception as exc:
        output = {"sufficient": True, "gaps": []}
        accepted, rejected = [], [{"question": "Final reflection", "reason": f"The evidence review could not be completed: {str(exc)[:180]}"}]

    unresolved = [*state.get("unresolved_gaps", []), *[item for item in rejected if item.get("unresolved") or not item.get("query")]]
    round_limit = state.get("iteration", 0) >= state.get("max_iterations", 2)
    if round_limit:
        unresolved.extend({"question": gap["question"], "reason": f"Unresolved after the configured {state.get('max_iterations', 2)} self-check passes: {gap.get('reason', 'more evidence was needed')}"} for gap in accepted)
        accepted = []
    enough = bool(output.get("sufficient", True)) and not accepted
    return {
        "gaps": accepted,
        "unresolved_gaps": unresolved,
        "reflection_sufficient": enough,
        "reflection_rejected": rejected,
        "current_step": f"Reflection found {len(accepted)} new gaps; {len(unresolved)} unresolved",
    }


def start_gap_research(state: dict[str, Any]) -> dict[str, Any]:
    """Queue novel gap queries and advance the bounded reflection loop."""
    gaps = state.get("gaps", [])
    questions = [{"question": gap["question"], "queries": gap["new_queries"]} for gap in gaps]
    names = {item["question"] for item in questions}
    unresolved = [item for item in state.get("unresolved_gaps", []) if item.get("question") not in names]
    return {
        "iteration": state.get("iteration", 0) + 1,
        "sub_questions": questions,
        "research_index": 0,
        "unresolved_gaps": unresolved,
        "query_retry_count": 0,
        "active_queries": [],
        "current_step": "Queued new gap questions",
    }


def comparer(state: dict[str, Any]) -> dict[str, Any]:
    """Compare source types, positions, agreements, and disagreements."""
    sources = state.get("sources", [])
    payload = {
        "findings": [{"question": item["question"], "summary": item["summary"][:800], "source_ids": item.get("source_ids", [])} for item in state.get("findings", [])],
        "sources": [{"id": source["id"], "title": source["title"], "domain": source.get("domain", ""), "type": source.get("type", "other"), "relevance": source.get("relevance_score", 2), "snippet": source.get("snippet", "")[:180], "text_excerpt": source.get("text", "")[:320]} for source in sources],
    }
    try:
        output = _structured(ComparisonOutput, prompts.COMPARER, json.dumps(payload, ensure_ascii=False), model_name=state.get("model"))
        comparison = output
        existing = {item.get("id") for item in comparison.get("sources", [])}
        for source in sources:
            if source["id"] not in existing:
                comparison["sources"].append({
                    "id": source["id"], "name": source.get("title", "Source"), "domain": source.get("domain", ""),
                    "type": source.get("type", "other"), "stance": "See associated finding.", "strengths": [], "caveats": [],
                })
    except Exception:
        comparison = {
            "sources": [{"id": source["id"], "name": source.get("title", "Source"), "domain": source.get("domain", ""), "type": source.get("type", "other"), "stance": "See associated finding.", "strengths": [], "caveats": []} for source in sources],
            "agreements": [], "disagreements": [],
        }
    return {"comparison": comparison, "current_step": "Compared source perspectives"}


def filter_verified_claims(
    claims: list[dict[str, Any]],
    finding_sources: dict[str, list[int]],
    valid_source_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep supported claims with valid citations; isolate all others."""
    supported: list[dict[str, Any]] = []
    not_established: list[dict[str, Any]] = []
    for item in claims:
        question = str(item.get("question", ""))
        allowed = set(finding_sources.get(question, [])) & valid_source_ids
        citations = sorted(set(item.get("supporting_source_ids", [])) & allowed)
        status = item.get("status")
        if status in {"supported", "partially supported"} and citations:
            supported.append({**item, "supporting_source_ids": citations})
        else:
            not_established.append({"question": question, "claim": item.get("claim", ""), "reason": item.get("rationale", "The retrieved text did not support this claim."), "source_ids": citations})
    return supported, not_established


def faithfulness_check(state: dict[str, Any]) -> dict[str, Any]:
    """Verify each draft claim against only its retrieved source excerpts."""
    sources = state.get("sources", [])
    source_by_id = {int(source["id"]): source for source in sources}
    finding_sources = {item["question"]: item.get("source_ids", []) for item in state.get("findings", [])}
    payload_findings = []
    for finding in state.get("findings", []):
        evidence = []
        for source_id in finding.get("source_ids", []):
            source = source_by_id.get(int(source_id))
            if source:
                evidence.append({"id": source_id, "type": source.get("type", "other"), "text": (source.get("text") or source.get("snippet", ""))[:650]})
        payload_findings.append({"question": finding["question"], "draft": finding["summary"][:1400], "evidence": evidence})
    try:
        output = _structured(FaithfulnessOutput, prompts.FAITHFULNESS, json.dumps({"findings": payload_findings}, ensure_ascii=False), model_name=state.get("model"))
        checked = output["claims"]
    except Exception as exc:
        checked = [{"question": item["question"], "claim": item["summary"], "rationale": f"Claim review could not run: {str(exc)[:120]}", "status": "unsupported", "supporting_source_ids": []} for item in state.get("findings", [])]
        rejected = []
    else:
        rejected = []
    supported, moved = filter_verified_claims(checked, finding_sources, set(source_by_id))
    not_established = [*state.get("not_established", []), *rejected, *moved]
    # If the verifier omitted an entire finding, do not silently present it.
    verified_questions = {item["question"] for item in checked}
    for finding in state.get("findings", []):
        if finding["question"] not in verified_questions:
            not_established.append({"question": finding["question"], "claim": finding["summary"], "reason": "The verifier returned no claim-level assessment."})
    return {"verified_findings": supported, "not_established": not_established, "current_step": f"Verified {len(supported)} claims; withheld {len(not_established)}"}


def _confidence(claim: dict[str, Any], sources_by_id: dict[int, dict[str, Any]]) -> str:
    """Estimate confidence from independent supporting source quality."""
    rows = [sources_by_id[item] for item in claim.get("supporting_source_ids", []) if item in sources_by_id]
    domains = {row.get("domain") for row in rows if row.get("domain")}
    strong = sum(row.get("type") in {"primary", "review"} for row in rows)
    if claim.get("status") == "supported" and len(domains) >= 3 and strong >= 2:
        return "High"
    if len(rows) >= 2 and strong >= 1:
        return "Medium"
    if len(rows) >= 2 and claim.get("status") == "supported":
        return "Medium"
    return "Low"


def _comparison_markdown(comparison: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    """Render one cited row per source with an explicit quality label."""
    by_id = {int(item["id"]): item for item in sources}
    assessments = {int(item["id"]): item for item in comparison.get("sources", []) if str(item.get("id", "")).isdigit()}
    lines = [
        "## Source Comparison",
        "",
        "| Evidence type / source | What it found | Strengths | Caveats |",
        "|---|---|---|---|",
    ]
    for source_id, source in by_id.items():
        assessment = assessments.get(source_id, {})
        name = str(assessment.get("name") or source.get("title") or source.get("domain") or "Source").replace("|", "\\|")
        domain = str(source.get("domain", "")).replace("|", "\\|")
        quality = str(source.get("type", "other")).replace("_", " ").title()
        stance = str(assessment.get("stance", "Relevant material retrieved.")).replace("|", "\\|")
        strengths = "; ".join(assessment.get("strengths", [])[:2]) or "Relevant to the question"
        caveats = "; ".join(assessment.get("caveats", [])[:2]) or "Interpret within the study context"
        strengths = strengths.replace("|", "\\|")
        caveats = caveats.replace("|", "\\|")
        label = f"{name} ({domain})" if domain and domain not in name else name
        lines.append(f"| {label} · {quality} | {stance} | {strengths} | {caveats} |")
    lines.append("")
    lines.append("*Source types are automated classifications from available page text and search descriptions.*")
    return "\n".join(lines)


def writer(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble a cited report from verified claims and comparison rows."""
    sources = state.get("sources", [])
    by_id = {int(source["id"]): source for source in sources}
    verified = state.get("verified_findings", [])
    narrative_payload = {
        "topic": state.get("topic", ""),
        "report_instructions": state.get("report_instructions", []),
        "verified_claims": [{"question": item["question"], "claim": item["claim"], "status": item["status"], "source_ids": item["supporting_source_ids"]} for item in verified],
        "agreements": state.get("comparison", {}).get("agreements", []),
        "disagreements": state.get("comparison", {}).get("disagreements", []),
        "unresolved_gaps": state.get("unresolved_gaps", []),
        "not_established": state.get("not_established", []),
    }
    try:
        narrative = _structured(ReportNarrative, prompts.WRITER, json.dumps(narrative_payload, ensure_ascii=False), model_name=state.get("model"))
    except Exception:
        narrative = {
            "title": state.get("topic", "Research Report"),
            "executive_summary": "The report includes only claims that passed source verification. Review the limitations for questions that remain unresolved.",
            "agreements_disagreements": "The available material did not support a reliable cross-source comparison.",
            "limitations": "Some claims or questions could not be established from the retrieved material.",
        }

    lines = [f"# {narrative['title']}", "", "## Executive Summary", "", narrative["executive_summary"].strip(), "", "## Key Findings", ""]
    if verified:
        for item in verified:
            confidence = _confidence(item, by_id)
            refs = "".join(f"[{source_id}]" for source_id in item["supporting_source_ids"])
            qualification = " (partially supported)" if item["status"] == "partially supported" else ""
            lines.append(f"- **{confidence} confidence{qualification}:** {item['claim'].strip()} {refs}")
    else:
        lines.append("No claim passed the source verification step for this run.")
    lines.extend(["", _comparison_markdown(state.get("comparison", {}), sources), "", "## Agreements vs Disagreements", "", narrative["agreements_disagreements"].strip()])
    agreement_points = state.get("comparison", {}).get("agreements", [])
    disagreement_points = state.get("comparison", {}).get("disagreements", [])
    if agreement_points or disagreement_points:
        if agreement_points:
            lines.extend(["", "**Agreement:**"])
            for point in agreement_points:
                refs = "".join(f"[{source_id}]" for source_id in point.get("source_ids", []))
                lines.append(f"- {point['text']} {refs}".rstrip())
        if disagreement_points:
            lines.extend(["", "**Disagreement:**"])
            for point in disagreement_points:
                refs = "".join(f"[{source_id}]" for source_id in point.get("source_ids", []))
                lines.append(f"- {point['text']} {refs}".rstrip())
    lines.extend(["", "## Open Questions/Limitations", "", narrative["limitations"].strip()])
    if state.get("unresolved_gaps"):
        lines.extend(["", "**Unresolved questions:**"])
        lines.extend(f"- **{item.get('question', 'Question')}:** {item.get('reason', 'The available search did not resolve this question.')}" for item in state["unresolved_gaps"])
    if state.get("not_established"):
        lines.extend(["", "### Not established by retrieved sources"])
        lines.extend(f"- {item.get('claim', '')} — {item.get('reason', 'Not supported by retrieved text.')}" for item in state["not_established"] if item.get("claim"))
    report = "\n".join(lines)
    source_lines = [f"[{item['id']}] [{item['title']}]({item['url']})" for item in sources]
    report += "\n\n## Sources\n\n" + ("\n".join(source_lines) if source_lines else "No relevant sources were retained.")
    report = normalize_citations(report, sources)
    return {"report": report, "current_step": "Report written"}


def route_after_research(state: dict[str, Any]) -> str:
    """Continue question-by-question, then send completed research to reflection."""
    return "researcher" if state.get("research_index", 0) < len(state.get("sub_questions", [])) else "reflector"


def route_after_relevance(state: dict[str, Any]) -> str:
    """Run one query rewrite when fewer than two results pass relevance."""
    return "query_rewriter" if state.get("needs_rewrite") and state.get("query_retry_count", 0) == 0 else "reader"


def route_after_query_rewrite(state: dict[str, Any]) -> str:
    """Search a novel rewrite, or continue with the remaining relevant results."""
    return "researcher" if state.get("rewrite_success") else "reader"


def route_after_reflection(state: dict[str, Any]) -> str:
    """Respect the configured self-check limit before comparison."""
    if not state.get("reflection_sufficient", True) and state.get("gaps") and state.get("iteration", 0) < state.get("max_iterations", 2):
        return "more_research"
    return "comparer"


# Fast path: one graph node maps all sub-questions concurrently. Legacy single-question
# helpers above remain available for compatibility with saved workflows/tests.
def _trim_relevant_text(text: str, question: str, limit: int = 1500) -> str:
    """Choose the most question-relevant paragraphs within a compact token budget."""
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    terms = _tokens(question)
    ranked = sorted(enumerate(paragraphs), key=lambda pair: (len(terms & _tokens(pair[1])), -pair[0]), reverse=True)
    chosen: list[tuple[int, str]] = []
    size = 0
    for index, paragraph in ranked:
        if len(terms & _tokens(paragraph)) == 0 and chosen:
            continue
        piece = paragraph[:limit]
        if size + len(piece) + 2 > limit:
            piece = piece[:max(0, limit - size - 2)]
        if piece:
            chosen.append((index, piece)); size += len(piece) + 2
        if size >= limit:
            break
    return "\n\n".join(item[1] for item in sorted(chosen))[:limit]


def _await_tasks(tasks: list[Any], timeout: float | None) -> list[Any]:
    """Wait boundedly for concurrent async jobs from synchronous graph nodes."""
    async def collect() -> list[Any]:
        if not tasks:
            return []
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        results = [task.result() for task in done if not task.cancelled() and task.exception() is None]
        for task in pending:
            task.cancel()
        return results
    return asyncio.run(collect())


async def _search_and_grade(question: str, queries: list[str], count: int, model: str, cache: dict[str, list[dict[str, Any]]], telemetry: Any, inflight: dict[str, asyncio.Task[list[dict[str, Any]]]], cache_lock: asyncio.Lock) -> tuple[str, list[dict[str, Any]]]:
    """Search one sub-question and grade results using the small helper model."""
    if telemetry:
        telemetry.record({"kind":"activity", "name":f"Searching: {question}", "duration_ms":0, "model":"", "prompt_chars":0, "approx_tokens":0, "retry_count":0, "rate_limited":False, "timeout":False, "error":""})
    async def one(query: str) -> list[dict[str, Any]]:
        async with cache_lock:
            if query in cache:
                return cache[query]
            task = inflight.get(query)
            if task is None:
                async def run_search() -> list[dict[str, Any]]:
                    try:
                        return await asyncio.to_thread(search, query, count)
                    except Exception:
                        return []
                task = asyncio.create_task(run_search())
                inflight[query] = task
        found = await task
        async with cache_lock:
            cache[query] = found
        return found
    raw_lists = await asyncio.gather(*(one(query) for query in queries))
    results = list({r["url"]: {**r, "query": q} for q, rows in zip(queries, raw_lists) for r in rows if r.get("url")}.values())
    if not results:
        return question, []
    payload = json.dumps({"question": question, "results": [{k: r.get(k, "") for k in ("title", "url", "snippet")} for r in results]}, ensure_ascii=False)
    try:
        output = await asyncio.to_thread(_structured, RelevanceOutput, prompts.RELEVANCE, payload, model, 1)
        kept, _ = filter_relevance_results(results, output.get("grades", []))
    except Exception:
        kept = _fallback_grade(question, results)
        kept, _ = filter_relevance_results(results, kept)
    if len(kept) < 2:
        # One small-model rewrite, only when fewer than two relevant results survive.
        try:
            rewrite_payload = json.dumps({"topic": question, "question": question, "queries_tried": queries}, ensure_ascii=False)
            rewrite = await asyncio.to_thread(_structured, QueryRewriteOutput, prompts.QUERY_REWRITE, rewrite_payload, model, 1)
            candidate = sanitize_search_query(rewrite["query"])
            if candidate and not is_duplicate_query(candidate, queries):
                rewritten = await one(candidate)
                combined = list({r["url"]: r for r in [*results, *[{**r, "query": candidate} for r in rewritten]] if r.get("url")}.values())
                grade = await asyncio.to_thread(_structured, RelevanceOutput, prompts.RELEVANCE, json.dumps({"question": question, "results": [{k:r.get(k, "") for k in ("title","url","snippet")} for r in combined]}, ensure_ascii=False), model, 1)
                kept, _ = filter_relevance_results(combined, grade.get("grades", []))
                queries.append(candidate)
        except Exception:
            pass
    if telemetry:
        telemetry.record({"kind":"activity", "name":f"Searched {question}", "duration_ms":0, "model":model, "prompt_chars":0, "approx_tokens":0, "retry_count":0, "rate_limited":False, "timeout":False, "error":"", "result_count":len(kept)})
    return question, kept


def researcher(state: dict[str, Any]) -> dict[str, Any]:
    """Search, grade, fetch, and summarize all questions concurrently."""
    telemetry = current_telemetry()
    questions = [dict(item) for item in state.get("sub_questions", [])]
    helper = state.get("helper_model", FALLBACK_MODEL)
    cache = dict(state.get("search_cache", {}))
    query_map = {str(item["question"]): list(item.get("queries", []))[:int(state.get("queries_per_question", 1))] for item in questions}
    async def search_all() -> list[Any]:
        inflight: dict[str, asyncio.Task[list[dict[str, Any]]]] = {}
        cache_lock = asyncio.Lock()
        return await asyncio.gather(*(_search_and_grade(q, query_map[q], int(state.get("results_per_query", 5)), helper, cache, telemetry, inflight, cache_lock) for q in query_map))
    remaining = telemetry.remaining() if telemetry else None
    try:
        searched = asyncio.run(asyncio.wait_for(search_all(), timeout=remaining))
    except (TimeoutError, asyncio.TimeoutError):
        searched = []
    results_by_question = dict(searched)
    page_limit = int(state.get("pages_per_question", 2))
    chosen: dict[str, dict[str, Any]] = {}
    for question, results in results_by_question.items():
        for result in results[:page_limit]:
            chosen.setdefault(result["url"], result)
    existing = list(state.get("sources", []))
    by_url = {item["url"]: item for item in existing}
    new_rows = [row for url, row in chosen.items() if url not in by_url]
    page_cache = dict(state.get("page_cache", {}))
    missing = [row["url"] for row in new_rows if row["url"] not in page_cache]
    try:
        fetched = asyncio.run(fetch_pages(missing, timeout=6, max_chars=3000)) if missing and not (telemetry and telemetry.expired()) else {}
    except Exception:
        fetched = {}
    page_cache.update(fetched)
    next_id = max([int(item.get("id", 0)) for item in existing] or [0]) + 1
    for result in new_rows:
        url = result["url"]
        source = {**result, "id": next_id, "domain": domain_name(url), "text": page_cache.get(url, ""), "type": result.get("source_type", "other")}
        next_id += 1
        existing.append(source); by_url[url] = source
    findings = list(state.get("findings", []))
    prior_qs = {f.get("question") for f in findings}
    pending = []
    for question in questions:
        q = str(question["question"])
        if q in prior_qs:
            continue
        evidence = []
        for row in results_by_question.get(q, [])[:page_limit]:
            source = by_url.get(row["url"])
            if not source:
                continue
            text = _trim_relevant_text(source.get("text", ""), q) or str(source.get("snippet", ""))[:650]
            if text:
                evidence.append({"id":source["id"], "title":source["title"], "type":source.get("type", "other"), "text":text})
        if evidence and not (telemetry and telemetry.expired()):
            p = json.dumps({"topic":state.get("topic", ""), "question":q, "evidence":evidence}, ensure_ascii=False)
            pending.append((q, evidence, p))
        else:
            findings.append({"question":q, "summary":"Retrieved search results did not provide enough readable material to answer this question.", "source_ids":[e["id"] for e in evidence], "search_queries":query_map.get(q, [])})
    async def summarize_all() -> list[Any]:
        return await asyncio.gather(*(asyncio.to_thread(_invoke_summary, payload, helper) for _, _, payload in pending))
    try:
        summaries = asyncio.run(asyncio.wait_for(summarize_all(), timeout=telemetry.remaining() if telemetry else None)) if pending else []
    except (TimeoutError, asyncio.TimeoutError):
        summaries = []
    for (q, evidence, _), summary in zip(pending, summaries):
        findings.append({"question":q, "summary":summary, "source_ids":[e["id"] for e in evidence], "search_queries":query_map.get(q, [])})
    for q, evidence, _ in pending[len(summaries):]:
        findings.append({"question":q, "summary":"The time budget ended before this sub-question could be summarized.", "source_ids":[e["id"] for e in evidence], "search_queries":query_map.get(q, [])})
    tried = list(state.get("queries_tried", []))
    for queries in query_map.values():
        tried.extend(queries)
    return {"findings":findings, "sources":existing, "page_cache":page_cache, "search_cache":cache, "queries_tried":list(dict.fromkeys(tried)), "time_expired":bool(telemetry and telemetry.expired()), "current_step":f"Researched {len(results_by_question)} sub-questions in parallel"}


def _invoke_summary(payload: str, model: str) -> str:
    """Summarize compact evidence with the fast helper model."""
    llm = make_llm(model)
    response = invoke_with_fallback(llm, [SystemMessage(content=prompts.SUMMARY), HumanMessage(content=payload)])
    return content_text(response)[:1200]


def reflector(state: dict[str, Any]) -> dict[str, Any]:
    """Check summaries for concrete new gaps with the fast model."""
    telemetry = current_telemetry()
    if state.get("time_expired") or (telemetry and telemetry.expired()) or int(state.get("max_iterations", 0)) <= int(state.get("iteration", 0)):
        return {"gaps":[], "reflection_sufficient":True, "current_step":"Reflection skipped: time or round limit reached"}
    payload = json.dumps({"topic":state.get("topic", ""), "findings":[{"question":f["question"],"summary":f["summary"][:350]} for f in state.get("findings", [])]}, ensure_ascii=False)
    try:
        result = _structured(ReflectionOutput, prompts.REFLECTOR, payload, state.get("helper_model", FALLBACK_MODEL), 1)
        accepted, _ = filter_novel_reflection_gaps(result.get("gaps", []), state.get("queries_tried", []))
    except Exception:
        accepted = []
    return {"gaps":accepted, "reflection_sufficient":not bool(accepted), "current_step":f"Self-check found {len(accepted)} specific new gaps"}


def comparer(state: dict[str, Any]) -> dict[str, Any]:
    """Compare short findings and source metadata with the selected large model."""
    sources = state.get("sources", [])
    payload = json.dumps({"findings":[{"question":f["question"],"summary":f["summary"][:300],"source_ids":f.get("source_ids", [])} for f in state.get("findings", [])], "sources":[{"id":s["id"],"title":s.get("title", ""),"domain":s.get("domain", ""),"type":s.get("type", "other"),"snippet":s.get("snippet", "")[:100]} for s in sources]}, ensure_ascii=False)
    try:
        comparison = _structured(ComparisonOutput, prompts.COMPARER, payload, state.get("model"), 1)
    except Exception:
        comparison = {"sources":[],"agreements":[],"disagreements":[]}
    return {"comparison":comparison, "current_step":"Compared source perspectives"}


def writer(state: dict[str, Any]) -> dict[str, Any]:
    """Write and faithfulness-check the report in one large-model call."""
    sources = state.get("sources", [])
    ids = {int(s["id"]) for s in sources}
    finding_sources = {f["question"]:f.get("source_ids", []) for f in state.get("findings", [])}
    payload = {"topic":state.get("topic", ""), "report_instructions":state.get("report_instructions", []),
        "findings":[{"question":f["question"],"summary":f["summary"][:450],"allowed_source_ids":f.get("source_ids", [])} for f in state.get("findings", [])],
        "source_catalog":[{"id":s["id"],"title":s.get("title", ""),"type":s.get("type", "other"),"snippet":s.get("snippet", "")[:100]} for s in sources],
        "comparison":state.get("comparison", {}), "unresolved_gaps":state.get("unresolved_gaps", [])}
    try:
        narrative = _structured(ReportNarrative, prompts.WRITER, json.dumps(payload, ensure_ascii=False), state.get("model"), 1)
    except Exception:
        narrative = {"title":state.get("topic", "Research Report"), "executive_summary":"This report summarizes the retrieved material; see findings and limitations for coverage.", "agreements_disagreements":"", "limitations":"Some questions may remain unresolved in the available source material.", "verified_findings":[]}
    def prose(value: Any) -> str:
        """Keep source IDs and inline evidence markers out of report prose."""
        text = str(value or "")
        text = re.sub(r"【\d+†[^】]*】", "", text)
        text = re.sub(r"(?<!\w)\[\d+\](?!\()", "", text)
        return re.sub(r"\s+", " ", text).strip()

    verified, not_established = filter_verified_claims(narrative.get("verified_findings", []), finding_sources, ids)
    by_id = {int(s["id"]):s for s in sources}
    lines = [f"# {prose(narrative['title'])}", "", "## Executive Summary", "", prose(narrative["executive_summary"]), "", "## Key Findings", ""]
    if verified:
        for claim in verified:
            lines.append(f"- **{_confidence(claim, by_id)} confidence:** {prose(claim['claim'])}")
    else:
        lines.extend(f"- **Low confidence:** {f['summary']}" for f in state.get("findings", []) if f.get("summary"))
    lines.extend(["", "## Agreements vs Disagreements", "", prose(narrative["agreements_disagreements"]), "", "## Open Questions/Limitations", "", prose(narrative["limitations"])])
    if state.get("unresolved_gaps"):
        lines.extend(["", *[f"- {g.get('question')}: {g.get('reason')}" for g in state["unresolved_gaps"]]])
    if not_established:
        lines.extend(["", *[f"- Not established: {g.get('claim')}" for g in not_established if g.get("claim")]])
    # The comparison is intentionally the final report section before the linked sources.
    lines.extend(["", _comparison_markdown(state.get("comparison", {}), sources)])
    body = re.sub(r"【\d+†[^】]*】", "", "\n".join(lines))
    body = re.sub(r"(?<!\w)\[\d+\](?!\()", "", body)
    report = normalize_citations(body, sources)
    return {"report":report, "verified_findings":verified, "not_established":not_established, "current_step":"Report written"}


def route_after_research(state: dict[str, Any]) -> str:
    """Proceed directly to reflection after the parallel research batch."""
    if state.get("time_expired"):
        return "writer"
    if state.get("max_iterations", 1) <= state.get("iteration", 0):
        return "comparer"
    return "reflector"


def route_after_reflection(state: dict[str, Any]) -> str:
    """Route only specific, novel gaps within the time and iteration budgets."""
    telemetry = current_telemetry()
    if state.get("time_expired") or (telemetry and telemetry.expired()):
        return "writer"
    if not state.get("reflection_sufficient", True) and state.get("gaps") and state.get("iteration", 0) < state.get("max_iterations", 1):
        return "more_research"
    return "comparer"
