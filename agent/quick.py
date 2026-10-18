"""Fast Quick-mode research pipeline, separate from the Deep workflow."""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
import time
import threading
from concurrent.futures import Future
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model, field_validator

from agent.llm import QUICK_FALLBACK_MODEL, content_text, invoke_with_fallback, make_llm, normalize_citations, parse_json_response
from agent.telemetry import current_telemetry
from agent.tools import domain_name, fetch_pages, search


class Plan(BaseModel):
    """Three focused questions and a research plus practitioner query for each."""
    sub_questions: list[str] = Field(min_length=3, max_length=3)
    queries: list[str] = Field(min_length=6, max_length=6)


class SourceSummary(BaseModel):
    """A short source-grounded summary and classification."""
    source_id: int
    summary: str = Field(min_length=80)
    source_type: str = Field(min_length=2)
    stance: str = Field(min_length=3)


class Claim(BaseModel):
    """A source-grounded claim scoped to its study, dataset, or setting."""
    text: str = Field(min_length=12)
    source_id: int
    scope: str = Field(min_length=3, max_length=180)


class Finding(BaseModel):
    """One report finding with confidence and supporting source ids."""
    text: str = Field(min_length=12)
    confidence: str
    source_ids: list[int] = Field(min_length=1)


class ComparisonRow(BaseModel):
    """A source comparison row tied to a retrieved source id."""
    id: int
    name: str = ""
    domain: str = ""
    type: str = "other"
    stance: str = Field(min_length=8, max_length=140)
    strengths: list[str] = Field(min_length=1, max_length=1)
    caveats: list[str] = Field(min_length=1, max_length=1)

    @field_validator("strengths", "caveats", mode="before")
    @classmethod
    def normalize_list_fields(cls, value: Any) -> Any:
        """Accept a single item or an empty string from JSON-mode models."""
        if value is None or value == "": return []
        return [value] if isinstance(value, str) else value


class ComparisonPoint(BaseModel):
    """A cited agreement or disagreement."""
    text: str = Field(min_length=20, max_length=220)
    source_ids: list[int] = Field(min_length=1)


class Extraction(BaseModel):
    """Grounded concise source summaries and claims."""
    summaries: list[SourceSummary] = Field(min_length=1)
    claims: list[Claim] = Field(min_length=2, max_length=3)


class Findings(BaseModel):
    """Cited executive overview and key findings."""
    executive_summary: str
    findings: list[Finding] = Field(min_length=5, max_length=7)


class Comparison(BaseModel):
    """Source table and cross-source comparison."""
    sources: list[ComparisonRow] = Field(min_length=1)
    agreements: list[ComparisonPoint] = Field(max_length=1)
    disagreements: list[ComparisonPoint] = Field(max_length=1)


class Limitations(BaseModel):
    """Evidence limitations and open questions."""
    limitations: list[str] = Field(max_length=2)
    open_questions: list[str] = Field(max_length=2)
    not_established: list[str] = Field(max_length=2)


class QuickStepFailure(RuntimeError):
    """A Quick-mode step failed after one shortened-prompt retry."""


def sync_node(function):
    """Adapt one async Quick node for the app's synchronous graph stream."""
    def run(state: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(function(state))
    run.__name__ = function.__name__
    return run


async def _daemon_call(function, *args):
    """Run one blocking search in a daemon thread without loop-owned executor waits."""
    context = contextvars.copy_context()
    result: Future[Any] = Future()
    def work() -> None:
        if not result.set_running_or_notify_cancel():
            return
        try:
            result.set_result(context.run(function, *args))
        except BaseException as exc:
            result.set_exception(exc)
    threading.Thread(target=work, name="quick-search", daemon=True).start()
    return await asyncio.wrap_future(result)


def _call_json(system: str, prompt: str, model: str, tokens: int, schema: type[BaseModel]) -> dict[str, Any]:
    """Request and validate JSON, retrying once with a compact prompt."""
    last_error: Exception | None = None
    for attempt in range(2):
        compact = prompt if attempt == 0 else _compact_prompt(prompt)
        try:
            schema_keys = ", ".join(schema.model_fields)
            max_tokens = tokens * (2 if attempt else 1)
            telemetry = current_telemetry()
            if telemetry and telemetry.mode == "Quick" and model == QUICK_FALLBACK_MODEL:
                max_tokens = min(max_tokens, 700 if attempt else 500)
            response = invoke_with_fallback(make_llm(model, max_tokens=max_tokens), [SystemMessage(content=system + f"\nReturn one JSON object only, with exactly these top-level keys: {schema_keys}. Do not return a schema description."), HumanMessage(content=compact)], json_mode=True, detect_truncation=True)
            data = parse_json_response(content_text(response))
            data = _coerce_schema_payload(data, schema)
            return schema.model_validate(data).model_dump()
        except Exception as exc:
            last_error = exc
    telemetry = current_telemetry()
    if telemetry:
        telemetry.record({"kind": "warning", "name": "step failed", "error": str(last_error)[:300]})
    raise QuickStepFailure(f"Step failed after one retry: {last_error}") from last_error


def _compact_prompt(prompt: str) -> str:
    """Shorten a retry prompt while preserving valid evidence JSON and output instructions."""
    marker = "Evidence JSON: "
    if marker in prompt:
        prefix, tail = prompt.split(marker, 1)
        payload, sep, instruction = tail.partition("\nReturn")
        if sep:
            try:
                evidence = json.loads(payload)
                claims = evidence.get("claims", [])[:8]
                compact_claims = [{"text": str(claim.get("text", ""))[:220], "source_id": claim.get("source_id"), "scope":str(claim.get("scope", ""))[:100]} for claim in claims]
                compact_sources = [{"id": item.get("id"), "title": str(item.get("title", ""))[:100], "domain": item.get("domain", ""), "type": item.get("type", "other")} for item in evidence.get("sources", [])[:6]]
                evidence = {"topic": evidence.get("topic", ""), "claims": compact_claims, "sources": compact_sources}
                return f"{prefix}{marker}{json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}\nReturn{instruction}\nKeep the JSON concise and valid."
            except (ValueError, TypeError):
                pass
    if "\nSources: " in prompt:
        prefix, tail = prompt.split("\nSources: ", 1)
        payload, sep, instruction = tail.partition("\nReturn JSON")
        if sep:
            try:
                rows = json.loads(payload)
                compact_rows = [{**row, "snippet":str(row.get("snippet", ""))[:180], "text":str(row.get("text", ""))[:750]} for row in rows[:3]]
                return f"{prefix}\nSources: {json.dumps(compact_rows, ensure_ascii=False, separators=(',', ':'))}\nReturn JSON{instruction}\nKeep summaries concise and the JSON valid."
            except (ValueError, TypeError):
                pass
    if len(prompt) <= 1800:
        return prompt + "\nReturn valid compact JSON only."
    return prompt[:850] + "\n[Evidence shortened for retry.]\n" + prompt[-850:] + "\nReturn valid compact JSON only."


def _coerce_schema_payload(data: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    """Normalize equivalent JSON shapes returned by different Groq models."""
    def as_text(value: Any) -> Any:
        return " ".join(str(part).strip() for part in value if str(part).strip()) if isinstance(value, list) else value
    if schema is Extraction:
        for item in data.get("summaries", []):
            if isinstance(item, dict):
                for key in ("summary", "source_type", "stance"):
                    if key in item: item[key] = as_text(item[key])
        for item in data.get("claims", []):
            if isinstance(item, dict):
                if "text" not in item and "claim" in item: item["text"] = item.pop("claim")
                if "text" in item: item["text"] = as_text(item["text"])
    elif schema is Findings:
        if "executive_summary" in data: data["executive_summary"] = as_text(data["executive_summary"])
        for item in data.get("findings", []):
            if isinstance(item, dict) and "text" in item: item["text"] = as_text(item["text"])
    elif issubclass(schema, Comparison):
        for key in ("agreements", "disagreements"):
            for item in data.get(key, []):
                if isinstance(item, dict):
                    if "text" not in item and "point" in item: item["text"] = item.pop("point")
                    if "text" in item: item["text"] = as_text(item["text"])
        for item in data.get("sources", []):
            if not isinstance(item, dict): continue
            if "stance" in item: item["stance"] = as_text(item["stance"])
            for key in ("strengths", "caveats"):
                if key in item:
                    values = item[key] if isinstance(item[key], list) else [item[key]]
                    item[key] = [str(as_text(value)) for value in values if str(as_text(value)).strip()]
    return data


def _overlap(text: str, query: str) -> float:
    """Calculate simple useful-word overlap for relevance ranking."""
    stop = {"what", "when", "which", "where", "with", "from", "that", "this", "into", "have", "does", "using", "about", "their", "between"}
    terms = {word.lower() for word in re.findall(r"[a-zA-Z0-9-]{3,}", query) if word.lower() not in stop}
    haystack = text.lower()
    return len([term for term in terms if term in haystack]) / max(1, len(terms))


def _relevant_text(text: str, query: str, limit: int = 1500) -> str:
    """Select paragraphs with highest question keyword overlap."""
    paras = [part.strip() for part in re.split(r"\n+", text) if len(part.strip()) > 35]
    ranked = sorted(enumerate(paras), key=lambda item: (_overlap(item[1], query), -item[0]), reverse=True)
    chosen: list[str] = []
    total = 0
    for _, paragraph in ranked:
        if total >= limit:
            break
        part = paragraph[: max(0, limit-total)]
        chosen.append(part)
        total += len(part)
    return "\n".join(chosen) or text[:limit]


async def plan(state: dict[str, Any]) -> dict[str, Any]:
    """Plan three questions and pair primary with practitioner searches."""
    topic = state["topic"]
    prompt = f"Research topic: {topic}\nDerive the dimensions implied by the question. For a 'differences between X and Y' question, cover practical trade-offs across cost, latency, knowledge updates, data requirements, hallucination risk, and maintenance; group related dimensions into exactly 3 focused questions. For technical topics, create TWO distinct keyword queries for EACH question: one targeting primary research/benchmarks and one targeting practitioner guides or official documentation. Return 3 sub_questions and 6 queries ordered as a research/practitioner pair for each question. Do not use restrictive site: filters. Return only JSON keys sub_questions and queries."
    data = await asyncio.to_thread(_call_json, "Plan concise web research. Return valid JSON only.", prompt, state["model"], 700, Plan)
    queries = list(data["queries"])
    return {"sub_questions": [{"question": q, "queries": [queries[i*2].strip(), queries[i*2+1].strip()]} for i, q in enumerate(data["sub_questions"])], "current_step": "Planned three focused questions with research and practitioner searches."}


def _domain_bonus(url: str) -> float:
    """Prefer primary research and reputable institutional hosts in Quick mode."""
    host = urlparse(url).netloc.lower()
    good = (".gov", ".edu", "arxiv.org", "pubmed", "nature.com", "science.org", "acm.org", "ieee.org", "doi.org")
    if any(item in host for item in good):
        return 0.45
    low_quality = ("geeksforgeeks.org", "onpilot.ai", "skilbrill.com", "medium.com")
    return -0.12 if any(item in host for item in low_quality) else 0.0


def _canonical_url(url: str) -> str:
    """Collapse arXiv format/version variants to one stable source link."""
    parsed = urlparse(url)
    if parsed.netloc.lower().endswith("arxiv.org"):
        match = re.search(r"/(?:abs|html|pdf)/([0-9.]+)(?:v\d+)?", parsed.path)
        if match:
            return f"https://arxiv.org/abs/{match.group(1)}"
    return url


def _source_key(source: dict[str, Any]) -> str:
    """Return a stable identity based on DOI, arXiv ID, or normalized title."""
    url = str(source.get("url", ""))
    doi = re.search(r"10\.\d{4,9}/[^\s?#]+", url, re.I)
    if doi:
        return "doi:" + doi.group(0).lower().rstrip(".")
    arxiv = re.search(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)(\d{4}\.\d{4,5})(?:v\d+)?", url, re.I)
    if arxiv:
        return "arxiv:" + arxiv.group(1)
    title = re.sub(r"[^a-z0-9]+", " ", str(source.get("title", "")).lower()).strip()
    return "title:" + title


def _dedupe_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge repeated papers and renumber the unique sources from one."""
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _source_key(row)
        if not key or key == "title:":
            key = "url:" + str(row.get("url", ""))
        if key not in merged:
            merged[key] = {**row, "questions": [row.get("question", "")]}
        else:
            current = merged[key]
            current["questions"] = list(dict.fromkeys(current.get("questions", []) + [row.get("question", "")]))
            for field in ("snippet", "text", "excerpt"):
                if len(str(row.get(field, ""))) > len(str(current.get(field, ""))):
                    current[field] = row[field]
    return [{**row, "id": index + 1} for index, row in enumerate(merged.values())]


async def research(state: dict[str, Any]) -> dict[str, Any]:
    """Search questions concurrently, rank results, and fetch within the early cutoff."""
    telemetry = current_telemetry()
    grace = max(0.0, float(state.get("search_grace_seconds", 10.0)))
    cutoff = (max(telemetry.started_at + telemetry.time_budget * 0.35, time.monotonic() + grace) if telemetry else time.monotonic() + 31.5)
    questions = state["sub_questions"]
    async def do_search(item: dict[str, Any], query: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            return item, await _daemon_call(search, query, 4)
        except Exception as exc:
            if telemetry: telemetry.record({"kind": "warning", "name": "search failed", "error": str(exc)[:250]})
            return item, []
    tasks = [asyncio.create_task(do_search(item, query)) for item in questions for query in item["queries"]]
    done, pending = await asyncio.wait(tasks, timeout=max(0, cutoff-time.monotonic()))
    for task in pending: task.cancel()
    search_rows = [task.result() for task in done if not task.cancelled()]
    gathered: list[dict[str, Any]] = []
    for item, results in search_rows:
        for row in results:
            row = dict(row)
            title = re.split(r"\.\.\.|…", str(row.get("title", "")), maxsplit=1)[0].strip()
            row["title"] = title[:180] or item["question"][:120]
            row["url"] = _canonical_url(str(row.get("url", "")))
            row["question"] = item["question"]
            row["relevance_score"] = round(_overlap(row.get("title", "")+" "+row.get("snippet", ""), item["question"])+_domain_bonus(row.get("url", "")), 3)
            gathered.append(row)
    # Keep relevant results across both research and practitioner queries.
    selected: list[dict[str, Any]] = []
    for item in questions:
        ranked = sorted((r for r in gathered if r["question"] == item["question"]), key=lambda r: r["relevance_score"], reverse=True)[:3]
        selected.extend(ranked)
    unique = _dedupe_sources(selected)
    sources = [{**row, "domain": domain_name(row["url"]), "text": ""} for row in unique]
    remaining = max(0.05, min(5.0, cutoff-time.monotonic()))
    try:
        texts = await asyncio.wait_for(fetch_pages([s["url"] for s in sources], timeout=5.0, max_chars=3000), timeout=remaining)
    except (TimeoutError, asyncio.TimeoutError):
        texts = {}
        if telemetry: telemetry.record({"kind": "warning", "name": "fetch cutoff", "error": "Search/fetch phase cutoff reached; continuing with snippets."})
    for source in sources:
        source["text"] = texts.get(source["url"], "")
        source["excerpt"] = _relevant_text(source["text"], source["question"]) if source["text"] else source.get("snippet", "")[:1500]
        host_title = (source["url"] + " " + source["title"]).lower()
        source["type"] = "documentation" if any(term in host_title for term in ("docs.", "documentation", "developer guide", "user guide")) else "practitioner" if any(term in host_title for term in ("guide", "blog", "engineering", "tutorial")) else "primary" if _domain_bonus(source["url"]) else "other"
    activity = f"Searched {len(questions)} questions; retained {len(sources)} sources."
    if telemetry: telemetry.record({"kind":"activity", "name":activity})
    return {"sources":sources, "current_step":activity}


async def extract(state: dict[str, Any]) -> dict[str, Any]:
    """Extract grounded claims in parallel, one small-model batch per question."""
    telemetry = current_telemetry()
    sources = state.get("sources", [])
    questions = state["sub_questions"]
    failed_questions: list[str] = []
    async def one(question: dict[str, Any]) -> dict[str, Any] | None:
        matched = [s for s in sources if question["question"] in s.get("questions", [s.get("question")])]
        if not matched: return None
        payload = [{"id":s["id"],"title":s["title"],"url":s["url"],"snippet":s.get("snippet",""),"text":s.get("excerpt","")} for s in matched]
        prompt = f"Question: {question['question']}\nSources (only these IDs): {json.dumps(payload, ensure_ascii=False)}\nReturn JSON: summaries, exactly one per source, with source_id, summary of exactly 3 short sentences (max 12 words each), source_type (max 4 words), stance (max 8 words); and exactly 2 concise claims for this question, each max 22 words, tied to source_id, and with scope naming the study, dataset, population, or setting (write 'general method description' only if scope is not narrower). Never turn one study's scoped result into a field-wide generalization. Use only supplied text/snippets; no unsupported absence claims."
        try:
            return await asyncio.to_thread(_call_json, "Extract facts faithfully from supplied source text and snippets. Cite only supplied source IDs.", prompt, state.get("helper_model", QUICK_FALLBACK_MODEL), 900, Extraction)
        except QuickStepFailure:
            failed_questions.append(question["question"])
            return None
    rows = await asyncio.gather(*(one(q) for q in questions))
    extractions = [row for row in rows if row]
    claims = [claim for row in extractions for claim in row.get("claims", [])]
    valid_ids = {int(s["id"]) for s in sources}
    claims = [dict(c, source_id=int(c["source_id"])) for c in claims if str(c.get("source_id", "")).isdigit() and int(c["source_id"]) in valid_ids]
    summaries = [item for row in extractions for item in row.get("summaries", [])]
    if not summaries or not claims:
        raise QuickStepFailure("No usable source summaries could be produced. Try another topic or search provider.")
    summarized_ids = {int(item["source_id"]) for item in summaries}
    missing_ids = sorted({int(s["id"]) for s in sources} - summarized_ids)
    if missing_ids:
        failed_questions.append("Some retrieved sources had no usable summary (source IDs " + ", ".join(map(str, missing_ids)) + ")")
        if telemetry:
            telemetry.record({"kind":"warning","name":"source extraction incomplete","error":"No usable summary for source IDs " + ", ".join(map(str, missing_ids))})
    return {"quick_claims":claims, "quick_summaries":summaries, "quick_extract_warnings":failed_questions, "current_step":f"Extracted grounded claims from {len(summaries)} source summaries."}


def _evidence(state: dict[str, Any]) -> str:
    """Serialize only structured claims and minimal source metadata for report calls."""
    source_by_id = {int(s["id"]):s for s in state.get("sources",[])}
    claims = [{"text":str(item.get("text", ""))[:180], "source_id":item.get("source_id"), "scope":str(item.get("scope", "unspecified scope"))[:120]} for item in state["quick_claims"][:12]]
    summary_by_id = {int(s["source_id"]):s for s in state.get("quick_summaries", [])}
    sources = [{"id":i,"title":s["title"][:120],"domain":s["domain"],"type":s.get("type","other"),"stance":summary_by_id.get(i,{}).get("stance","")} for i,s in source_by_id.items()]
    return json.dumps({"topic":state["topic"],"claims":claims,"sources":sources}, ensure_ascii=False, separators=(",", ":"))


def confidence_for_sources(source_ids: Any) -> str:
    """Map independent supporting source count to the fixed confidence rubric."""
    count = len(_ids(source_ids))
    return "High" if count >= 3 else "Medium" if count == 2 else "Low"


async def write_findings(state: dict[str, Any]) -> dict[str, Any]:
    """Write executive summary and five to seven cited findings."""
    prompt = f"Evidence JSON: {_evidence(state)}\nReturn JSON with a 60-80 word executive_summary and exactly 5 distinct findings. Each finding text is at most 22 words, confidence, source_ids. Cite every statement. Derive all claims only from supplied claims. Preserve each claim's scope (study/dataset/setting) in its wording; executive-summary statements must be supported across sources, and narrow single-study findings belong only in Key Findings with explicit scope. Never generalize one study to the whole field. Keep JSON compact."
    valid = {int(s["id"]) for s in state.get("sources",[])}
    try:
        data = await asyncio.to_thread(_call_json, "Write concise evidence-bound research synthesis. Every claim must cite supplied source IDs.", prompt, state["model"], 1800, Findings)
    except QuickStepFailure:
        claims = [c for c in state.get("quick_claims", []) if int(c.get("source_id", 0)) in valid]
        grounded: list[dict[str, Any]] = [{"text":str(c["text"]),"source_id":int(c["source_id"])} for c in claims]
        for summary in state.get("quick_summaries", []):
            source_id = int(summary.get("source_id", 0))
            if source_id not in valid: continue
            sentences = re.split(r"(?<=[.!?])\s+", str(summary.get("summary", "")))
            grounded.extend({"text":sentence.strip(),"source_id":source_id} for sentence in sentences if len(sentence.strip()) >= 20)
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in grounded:
            key = re.sub(r"\W+", " ", item["text"]).strip().lower()
            if key and key not in seen:
                unique.append(item); seen.add(key)
        if len(unique) < 5:
            raise
        selected = unique[:7]
        data = {"executive_summary":" ".join(f"{c['text']} [{c['source_id']}]" for c in selected[:2]), "findings":[{"text":c["text"],"confidence":"Low","source_ids":[int(c["source_id"])]} for c in selected]}
        return {"quick_findings":data, "quick_findings_failed":"Step failed after one retry; findings are shown directly from extracted claims."}
    data["findings"] = [f for f in data["findings"] if _ids(f.get("source_ids")) & valid]
    if len(data["findings"]) < 5: raise QuickStepFailure("Writer returned fewer than five source-backed findings.")
    for finding in data["findings"]:
        finding["source_ids"] = sorted(_ids(finding.get("source_ids")) & valid)
        finding["confidence"] = confidence_for_sources(finding["source_ids"])
    return {"quick_findings":data}


async def write_comparison(state: dict[str, Any]) -> dict[str, Any]:
    """Compare evidence-bearing sources and their claims."""
    source_ids = [int(s["id"]) for s in state.get("sources", [])]
    exact_comparison = create_model("ExactQuickComparison", __base__=Comparison, sources=(list[ComparisonRow], Field(min_length=len(source_ids), max_length=len(source_ids))))
    prompt = f"Evidence JSON: {_evidence(state)}\nActively compare claim text, scope, and each source stance: identify shared themes even when complementary, and tensions/conflicts even when studies use different settings. Distinguish complementarity from contradiction. Return one row for each source id {source_ids}. Each row: id, stance (max 8 words), strengths (one item max 8 words), caveats (one item max 8 words). Add agreements/disagreements only when grounded in the claims and cite involved source_ids. If none, use an empty array; code will explain why none were identified without citations. Points max 16 words. Use supplied claims only; compact valid JSON."
    valid = {int(s["id"]) for s in state.get("sources",[])}
    try:
        data = await asyncio.to_thread(_call_json, "Compare sources in compact evidence-bound JSON. Return valid JSON only.", prompt, state["model"], 1200, exact_comparison)
    except QuickStepFailure:
        return {"quick_comparison_failed":"Step failed after one retry; the comparison table will use extracted source summaries."}
    data["sources"] = [x for x in data["sources"] if int(x.get("id",0)) in valid]
    source_by_id = {int(s["id"]): s for s in state.get("sources", [])}
    for row in data["sources"]:
        source = source_by_id[int(row["id"])]
        row["name"] = str(source.get("title") or row.get("name") or f"Source {row['id']}")[:160]
        row["domain"] = str(source.get("domain") or row.get("domain") or "unknown")
        row["type"] = str(source.get("type") or row.get("type") or "other")
    def grounded_points(items):
        return [x for x in items if _ids(x.get("source_ids")) & valid and not re.search(r"\b(?:none identified|no agreement|no disagreement|no conflict)\b", str(x.get("text", "")), re.I)]
    data["agreements"] = grounded_points(data["agreements"])
    data["disagreements"] = grounded_points(data["disagreements"])
    if not data["sources"]: raise QuickStepFailure("Comparison step returned no valid source rows.")
    return {"quick_comparison":data}


async def write_limitations(state: dict[str, Any]) -> dict[str, Any]:
    """Identify limits and open questions grounded in the evidence."""
    prompt = f"Evidence JSON: {_evidence(state)}\nReturn JSON limitations, open_questions, not_established. Include at most one concise item in each array, max 16 words, each with numeric source_ids in brackets. Use supplied claims only. Never claim no evidence where claims cover the subject."
    try:
        data = await asyncio.to_thread(_call_json, "Describe evidence limits precisely in compact JSON. Never claim no evidence if claims exist.", prompt, state["model"], 400, Limitations)
    except QuickStepFailure:
        return {"quick_limitations_failed":"Step failed after one retry; no generated limitations are presented for this run."}
    return {"quick_limitations":data}


def _ids(value: Any) -> set[int]:
    """Extract valid numeric citation ids from model output."""
    try: return {int(x) for x in value}
    except (TypeError, ValueError): return set()


def _citation(ids: Any, valid: set[int]) -> str:
    """Format source ids as inline numeric citations."""
    return " ".join(f"[{i}]" for i in sorted(_ids(ids) & valid))


def _clean_table_citations(text: str, valid: set[int]) -> str:
    """Keep numeric source references in the Source cell, not repeated across columns."""
    rows = []
    citation = r"\[(\d+)\](?:\(\s*<https?://[^>]+>\s*\))?"
    for line in text.splitlines():
        if line.startswith("|") and line.count("|") >= 6:
            cells = line.split("|")
            for index in range(2, len(cells) - 1):
                cells[index] = re.sub(r"\s*" + citation, "", cells[index])
            rows.append("|".join(cells))
            continue
        chunks = re.split(r"(?<=[.!?])\s+", line)
        cleaned = []
        for chunk in chunks:
            seen: set[int] = set()
            def keep_once(match: re.Match[str]) -> str:
                sid = int(match.group(1))
                if sid not in valid or sid in seen:
                    return ""
                seen.add(sid)
                return f"[{sid}]"
            cleaned.append(re.sub(citation, keep_once, chunk))
        rows.append(" ".join(cleaned))
    return "\n".join(rows)


def _best_claim_ids(text: str, state: dict[str, Any]) -> set[int]:
    """Find the closest source-backed claim ids for a report sentence."""
    words = set(re.findall(r"[a-z0-9]{4,}", text.lower()))
    scored = []
    for claim in state.get("quick_claims", []):
        claim_words = set(re.findall(r"[a-z0-9]{4,}", str(claim.get("text", "")).lower()))
        overlap = len(words & claim_words)
        if overlap:
            scored.append((overlap, int(claim.get("source_id", 0))))
    valid = {int(s["id"]) for s in state.get("sources", [])}
    return {sid for _, sid in sorted(scored, reverse=True)[:2] if sid in valid}


def _cite_summary(summary: str, state: dict[str, Any]) -> str:
    """Normalize model citations and ensure every summary sentence has source ids."""
    valid = {int(s["id"]) for s in state.get("sources", [])}
    summary = re.sub(r"\[(\d+)\]\(\s*<https?://[^>]+>\s*\)", r"[\1]", summary)
    summary = _normalize_source_tags(summary, valid)
    summary = _normalize_parenthetical_citations(summary, valid)
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    rendered = []
    for sentence in sentences:
        ids = {int(n) for n in re.findall(r"\[(\d+)\]", sentence) if int(n) in valid}
        if not ids:
            ids = _best_claim_ids(sentence, state) or valid
        seen_ids: set[int] = set()
        def keep_first(match: re.Match[str]) -> str:
            sid = int(match.group(1))
            if sid not in valid or sid in seen_ids:
                return ""
            seen_ids.add(sid)
            return f"[{sid}]"
        sentence = re.sub(r"\[(\d+)\]", keep_first, sentence).strip()
        present = {int(n) for n in re.findall(r"\[(\d+)\]", sentence) if int(n) in valid}
        missing = ids - present
        if missing:
            sentence = sentence.rstrip() + " " + _citation(missing, valid)
        rendered.append(sentence)
    return " ".join(rendered)


def _scope_summary(summary: str, state: dict[str, Any]) -> str:
    """Explicitly scope executive-summary sentences grounded in one study."""
    claims = state.get("quick_claims", [])
    result = []
    for sentence in re.split(r"(?<=[.!?])\s+", summary.strip()):
        ids = {int(value) for value in re.findall(r"\[(\d+)\]", sentence)}
        if len(ids) == 1:
            sid = next(iter(ids))
            words = set(re.findall(r"[a-z0-9]{4,}", sentence.lower()))
            matches = [claim for claim in claims if int(claim.get("source_id", 0)) == sid and len(words & set(re.findall(r"[a-z0-9]{4,}", str(claim.get("text", "")).lower()))) >= 2]
            scope = str(matches[0].get("scope", "")).strip() if matches else ""
            if scope and scope.lower() not in {"general method description", "general", "not specified"} and not re.match(r"(?i)^(?:in|for|within)\b", sentence):
                sentence = f"In {scope}, {sentence[0].lower() + sentence[1:] if sentence else sentence}"
        result.append(sentence)
    return " ".join(result)


def _normalize_source_tags(text: str, valid: set[int]) -> str:
    """Convert prose source labels into valid numeric citations."""
    def replace(match: re.Match[str]) -> str:
        source_id = int(match.group(1))
        return f"[{source_id}]" if source_id in valid else ""
    text = re.sub(r"\bsource\s*#?\s*(\d+)\b", replace, text, flags=re.I)
    return re.sub(r"\(\s*(\[\d+\])\s*\)", r"\1", text)


def _normalize_citation_groups(text: str, valid: set[int]) -> str:
    """Expand grouped numeric citations into individual bracket citations."""
    text = re.sub(r"\[(\s*\d+(?:\s*[,;]\s*\d+)+\s*)\]", lambda m: "".join(f"[{n.strip()}]" for n in re.split(r"[,;]",m.group(1)) if n.strip().isdigit() and int(n.strip()) in valid), text)
    text = re.sub(r"(?<![\w\[])((?:\d+\s*[,;]\s*)+\d+)(?=[.,])", lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+",m.group(1)) if int(n) in valid), text)
    def split_concatenated(match: re.Match[str]) -> str:
        digits = match.group(1)
        return "".join(f"[{digit}]" for digit in digits) if all(int(digit) in valid for digit in digits) else match.group(0)
    return re.sub(r"(?<!\d)(\d{2})(?=\.)", split_concatenated, text)


def _normalize_parenthetical_citations(text: str, valid: set[int]) -> str:
    """Convert numeric citation groups, including partly linked groups, to markers."""
    text = re.sub(r"\(\s*\[(\d+)\]\s*\(<[^>]+>\)\s*[,;]\s*(\d+)\s*\)", lambda m: "".join(f"[{n}]" for n in (m.group(1),m.group(2)) if int(n) in valid), text)
    pattern = r"\(\s*((?:(?:\[\d+\]|\d+)\s*[,;]\s*)+(?:\[\d+\]|\d+))\s*\)"
    def replace(match: re.Match[str]) -> str:
        return "".join(f"[{i}]" for i in (int(n) for n in re.findall(r"\d+", match.group(1))) if i in valid)
    return re.sub(pattern, replace, text)


def _cite_evidence_note(text: str, state: dict[str, Any]) -> str:
    """Attach the closest extracted-claim IDs to limitation statements."""
    valid = {int(s["id"]) for s in state.get("sources", [])}
    text = _normalize_source_tags(text, valid)
    if any(int(item) in valid for item in re.findall(r"\[(\d+)\]", text)):
        return text
    words = {w for w in re.findall(r"[a-z0-9]{4,}", text.lower())}
    scored = []
    for claim in state.get("quick_claims", []):
        claim_words = set(re.findall(r"[a-z0-9]{4,}", str(claim.get("text", "")).lower()))
        overlap = len(words & claim_words)
        if overlap:
            scored.append((overlap, int(claim.get("source_id", 0))))
    selected = {sid for _, sid in sorted(scored, reverse=True)[:2] if sid in {int(s["id"]) for s in state.get("sources", [])}}
    return f"{text} {_citation(selected or {int(s['id']) for s in state.get('sources', [])}, {int(s['id']) for s in state.get('sources', [])})}".strip()


def _sanitize_limitations(state: dict[str, Any]) -> dict[str, list[str]]:
    """Remove contradicted absence claims and duplicate claims from unestablished items."""
    raw = state.get("quick_limitations", {})
    claims = state.get("quick_claims", [])
    valid_ids = {int(s["id"]) for s in state.get("sources", [])}
    def terms(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{4,}", value.lower()))
    def closest(value: str) -> tuple[dict[str, Any] | None, float]:
        words = terms(value)
        scored = [(len(words & terms(str(c.get("text", "")))) / max(1, len(words)), c) for c in claims]
        score, claim = max(scored, key=lambda x:x[0], default=(0.0, None))
        return claim, score
    limits = list(raw.get("limitations", []))
    clean_limits: list[str] = []
    for item in limits:
        item = str(item)
        claim, overlap = closest(item)
        absence = re.search(r"(?:no evidence(?: is)? provided|no evidence on|no data(?: is)? provided)\s*(?:regarding|on|for)?\s*(.+)", item, re.I)
        if absence:
            if claim and overlap >= 0.25:
                continue
            detail = absence.group(1).rstrip(" .")
            item = f"The extracted claims do not quantify {detail} under a directly comparable evaluation."
        elif re.search(r"absence of data limits conclusions", item, re.I):
            item = "The extracted claims do not include a directly comparable measurement for this question."
        elif claim and overlap >= 0.30 and re.search(r"\b(?:does not address|does not cover|lacks details|evidence only covers|no quantitative|provides no evidence)\b", item, re.I):
            continue
        elif claim and overlap >= 0.25 and re.search(r"\b(not provided|does not provide|is unavailable)\b", item, re.I):
            item = "The sources discuss related aspects but do not establish a directly comparable result for this question."
        clean_limits.append(_cite_evidence_note(item, state))
    clean_unestablished: list[str] = []
    for item in raw.get("not_established", []):
        claim, overlap = closest(str(item))
        if overlap < 0.55:
            clean_unestablished.append(_cite_evidence_note(str(item), state))
    source_types = {str(source.get("type", "")).lower() for source in state.get("sources", [])}
    source_types.update(str(item.get("source_type", "")).lower() for item in state.get("quick_summaries", []))
    claim_words = " ".join(str(item.get("text", "")).lower() for item in claims)
    if not source_types.intersection({"practitioner", "documentation", "guide", "docs"}):
        clean_limits.append(_cite_evidence_note("The retrieved source set lacks practitioner or official documentation perspectives.", state))
    if not re.search(r"\b(?:cost|latency|runtime|response time|price|expense)\b", claim_words):
        clean_limits.append(_cite_evidence_note("The retrieved claims do not provide cost or latency evidence.", state))
    clean_open = [str(item).strip() for item in raw.get("open_questions", []) if str(item).strip()]
    return {"limitations":clean_limits,"open_questions":clean_open,"not_established":clean_unestablished}


def fallback_comparison(state: dict[str, Any]) -> dict[str, Any]:
    """Build a useful cited comparison directly from extracted evidence."""
    summaries = {int(item["source_id"]): item for item in state.get("quick_summaries", [])}
    claims: dict[int, list[str]] = {}
    for item in state.get("quick_claims", []):
        sid = int(item.get("source_id", 0))
        claims.setdefault(sid, []).append(str(item.get("text", "")).strip())
    rows = []
    for source in state.get("sources", []):
        sid = int(source["id"])
        summary = summaries.get(sid, {})
        support = str(summary.get("summary", "")).strip()
        if not support:
            support = str(source.get("excerpt") or source.get("snippet") or "Retrieved metadata only; no source text was available.").strip()
        stance = str(summary.get("stance") or (claims.get(sid) or ["The extracted claim describes this source's reported result."])[0])
        caveat = "Source text was unavailable; this row uses the retrieved search excerpt." if not source.get("text") else "The comparison is limited to the text retrieved for this run."
        rows.append({"id": sid, "name": source.get("title", f"Source {sid}"), "domain": source.get("domain", ""),
                     "type": summary.get("source_type") or source.get("type", "other"), "stance": stance,
                     "strengths": [support], "caveats": [caveat]})
    ids = [int(source["id"]) for source in state.get("sources", [])]
    cite = ids[:]
    return {"sources": rows, "agreements": [], "disagreements": []}


def fallback_limitations(state: dict[str, Any]) -> dict[str, list[str]]:
    """State concrete study-scope limits from the retrieved evidence."""
    ids = [int(source["id"]) for source in state.get("sources", [])]
    cite = _citation(ids, set(ids))
    snippet_ids = [int(source["id"]) for source in state.get("sources", []) if not source.get("text")]
    limits = []
    if snippet_ids:
        limits.append("Full page text was unavailable for some sources, so those entries rely on search excerpts " + _citation(snippet_ids, set(ids)) + ".")
    return {
        "limitations": limits or ["The synthesis is limited to the retrieved studies and the claims extracted from them " + cite + "."],
        "open_questions": ["A matched domain-QA evaluation comparing answer quality, freshness, latency, and total cost remains open " + cite + "."],
        "not_established": [],
    }


def assemble(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble Markdown and render the comparison table in code."""
    sources = state.get("sources", [])
    valid = {int(s["id"]) for s in sources}
    findings = state["quick_findings"]
    comparison_failed = state.get("quick_comparison_failed", "")
    comparison = fallback_comparison(state) if comparison_failed else state["quick_comparison"]
    limits_failed = state.get("quick_limitations_failed", "")
    limits = fallback_limitations(state) if limits_failed else _sanitize_limitations(state)
    summary = _scope_summary(_cite_summary(findings["executive_summary"], state), state)
    lines = [f"# {state['topic']}"]
    if state.get("quick_extract_warnings"):
        question_list = "; ".join(state["quick_extract_warnings"])
        lines.extend(["", f"> Extraction step failed for these questions: {question_list}"])
    lines.extend(["", "## Executive Summary", "", summary, "", "## Key Findings", ""])
    for finding in findings["findings"]:
        cite = _citation(finding.get("source_ids"), valid)
        lines.append(f"- {finding['text']} **Confidence: {finding.get('confidence','Medium')}.** {cite}".rstrip())
    lines.extend(["", "## Agreements vs Disagreements", ""])
    agreements = comparison.get("agreements", [])
    disagreements = comparison.get("disagreements", [])
    if agreements:
        lines.append("**Agreements**")
        lines.extend(f"- {x['text']} {_citation(x.get('source_ids'), valid)}" for x in agreements)
    else:
        lines.append("**Agreements:** None identified because the retrieved claims share no clearly supported conclusion.")
    if disagreements:
        lines.append("\n**Disagreements**")
        lines.extend(f"- {x['text']} {_citation(x.get('source_ids'), valid)}" for x in disagreements)
    else:
        lines.append("\n**Disagreements:** None identified because the retrieved claims do not directly conflict or show a supported tension.")
    lines.extend(["", "## Open Questions/Limitations", ""])
    for label, key in (("Limitations", "limitations"), ("Open questions", "open_questions"), ("Not established by sources", "not_established")):
        vals = limits.get(key, [])
        if vals:
            lines.append(f"**{label}**")
            lines.extend(f"- {str(item).strip()}" if key == "open_questions" else f"- {_cite_evidence_note(str(item), state)}" for item in vals)
    lines.extend(["", "## Source Comparison", "", "| Source | Type | Stance | Strengths | Caveats |", "|---|---|---|---|---|"])
    by_id = {int(s["id"]):s for s in sources}
    for row in comparison["sources"]:
        sid = int(row["id"])
        name = row.get("name") or by_id[sid].get("title", f"Source {sid}")
        cell = lambda val: str(val or "—").replace("|", "\\|").replace("\n", " ")
        strengths = "; ".join(f"{item} [{sid}]" for item in row.get("strengths", []))
        caveats = "; ".join(f"{item} [{sid}]" for item in row.get("caveats", []))
        lines.append(f"| {cell(name)} [{sid}] | {cell(row.get('type'))} | {cell(row.get('stance'))} | {cell(strengths)} | {cell(caveats)} |")
    raw_report = _normalize_source_tags("\n".join(lines), valid)
    raw_report = _normalize_citation_groups(raw_report, valid)
    raw_report = _normalize_parenthetical_citations(raw_report, valid)
    raw_report = _clean_table_citations(raw_report, valid)
    report = normalize_citations(raw_report, sources)
    return {"report":report, "current_step":"Assembled the cited report."}
