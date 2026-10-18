"""Groq chat model setup and resilient invocation helpers."""

from __future__ import annotations

import os
import re
import time
import math
import threading
import uuid
from collections import deque
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agent.telemetry import current_telemetry

DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
QUICK_FALLBACK_MODEL = "qwen/qwen3.8-27b"
MODEL_ALIASES = {
    "llama-3.3-70b-versatile": DEFAULT_MODEL,
    "llama-3.1-8b-instant": FALLBACK_MODEL,
    "openai/gpt-oss-20b": FALLBACK_MODEL,
}


class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not configured."""


class _MinuteBudget:
    """Serialize estimated token reservations against a rolling TPM budget."""

    def __init__(self) -> None:
        self.limit = max(1000, int(os.getenv("GROQ_TPM_BUDGET", "6500")))
        self.condition = threading.Condition()
        self.usage: deque[tuple[float, int]] = deque()

    def reserve(self, tokens: int) -> None:
        """Wait until the request fits the configured rolling token budget."""
        tokens = min(tokens, self.limit)
        telemetry = current_telemetry()
        while True:
            with self.condition:
                now = time.monotonic()
                while self.usage and now - self.usage[0][0] >= 60:
                    self.usage.popleft()
                used = sum(count for _, count in self.usage)
                if used + tokens <= self.limit:
                    self.usage.append((now, tokens))
                    return
                delay = max(0.05, 60 - (now - self.usage[0][0]))
                if telemetry and delay > telemetry.remaining():
                    raise TimeoutError("Run time budget would expire while waiting for Groq token capacity.")
                started = time.monotonic()
                self.condition.wait(timeout=delay)
                waited = time.monotonic() - started
            if telemetry and waited:
                telemetry.record_sleep(waited, "tpm_budget")


_TPM_BUDGET = _MinuteBudget()


class _HeaderRateTracker:
    """Track only explicit per-model rate-limit headers; unknown limits never pace calls."""
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.models: dict[str, tuple[int, int, float]] = {}
        self.call_locks: dict[str, threading.Lock] = {}
        self.unavailable_until: dict[str, float] = {}

    def model_lock(self, model: str) -> threading.Lock:
        """Serialize brief header checks without serializing parallel model requests."""
        with self.lock:
            return self.call_locks.setdefault(model, threading.Lock())

    def update(self, model: str, headers: Any) -> None:
        """Remember reported remaining tokens, limit, and reset time for one model."""
        try:
            remaining = int(headers.get("x-ratelimit-remaining-tokens"))
            limit = int(headers.get("x-ratelimit-limit-tokens", remaining))
            reset = str(headers.get("x-ratelimit-reset-tokens", "0s"))
            match = re.match(r"([0-9.]+)(ms|s|m)?", reset)
            seconds = float(match.group(1)) if match else 0.0
            if match and match.group(2) == "ms": seconds /= 1000
            elif match and match.group(2) == "m": seconds *= 60
            with self.lock:
                self.models[model] = (remaining, limit, time.monotonic() + seconds)
        except (TypeError, ValueError):
            return

    def mark_unavailable(self, model: str, seconds: float) -> None:
        """Remember an observed daily quota reset so later calls can skip that model."""
        with self.lock:
            self.unavailable_until[model] = time.monotonic() + max(0.0, seconds)

    def is_unavailable(self, model: str) -> bool:
        """Check whether a prior provider quota response still blocks this model."""
        with self.lock:
            expiry = self.unavailable_until.get(model, 0.0)
            if expiry <= time.monotonic():
                self.unavailable_until.pop(model, None)
                return False
            return True

    def wait_if_needed(self, model: str, estimated_tokens: int, max_wait: float = 15.0) -> float:
        """Wait only when a prior real header proves the next request will not fit."""
        with self.lock:
            item = self.models.get(model)
        if not item:
            return 0.0
        remaining, _, reset_at = item
        if remaining >= estimated_tokens or reset_at <= time.monotonic():
            return 0.0
        delay = min(max(0.0, reset_at - time.monotonic()), max_wait)
        if delay:
            time.sleep(delay)
        return delay


_HEADER_RATE_TRACKER = _HeaderRateTracker()


def make_llm(model: str | None = None, max_tokens: int = 650) -> ChatGroq:
    """Create a ChatGroq client using environment configuration."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise MissingAPIKeyError("GROQ_API_KEY is missing. Add it to your .env file.")
    selected_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    selected_model = MODEL_ALIASES.get(selected_model, selected_model)
    client = ChatGroq(model=selected_model, api_key=key, temperature=0, timeout=20, max_tokens=max_tokens)
    _attach_rate_header_tracking(client, selected_model)
    return client


def _attach_rate_header_tracking(llm: ChatGroq, model: str) -> None:
    """Feed Groq response rate-limit headers into the per-model tracker."""
    try:
        transport = llm.client._client._client
        hooks = transport.event_hooks.setdefault("response", [])
        hooks.append(lambda response: _HEADER_RATE_TRACKER.update(model, response.headers))
    except (AttributeError, TypeError):
        # ChatGroq test doubles and future transports may not expose httpx hooks.
        return


def invoke_with_fallback(llm: ChatGroq, messages: list[Any], json_mode: bool = False, detect_truncation: bool = False) -> Any:
    """Call the LLM with TPM pacing, bounded 429 retries, and small-model fallback."""
    active_model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    prompt = "\n".join(str(getattr(message, "content", "")) for message in messages)
    prompt_chars = len(prompt)
    completion_budget = int(getattr(llm, "max_tokens", 650) or 650)
    estimate = math.ceil(prompt_chars / 4) + completion_budget
    telemetry = current_telemetry()
    quick_mode = bool(telemetry and telemetry.mode == "Quick")
    call_id = uuid.uuid4().hex[:10]
    started = time.monotonic()
    retries = 0
    retry_after_seconds = 0.0
    saw_429 = False
    rate_limit_count = 0
    tpm_limit_count = 0
    attempt_count = 0
    primary_skipped = False
    saw_timeout = False
    error = ""
    invoke_options = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        if not quick_mode:
            _TPM_BUDGET.reserve(estimate)
        while True:
            if quick_mode and _HEADER_RATE_TRACKER.is_unavailable(str(active_model)):
                primary_skipped = True
                break
            try:
                if quick_mode:
                    with _HEADER_RATE_TRACKER.model_lock(str(active_model)):
                        waited = _HEADER_RATE_TRACKER.wait_if_needed(str(active_model), estimate)
                        if telemetry and waited:
                            telemetry.record_sleep(waited, "header_token_wait", str(active_model), call_id)
                    # Release the gate before network I/O: parallel graph branches
                    # remain parallel, while later phases can use observed headers.
                    attempt_count += 1
                    result = llm.invoke(messages, **invoke_options)
                else:
                    attempt_count += 1
                    result = llm.invoke(messages, **invoke_options)
                finish = (getattr(result, "response_metadata", {}) or {}).get("finish_reason", "")
                if detect_truncation and finish == "length":
                    raise TruncatedOutputError("LLM response ended at the token limit (finish_reason=length).")
                return result
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                text = str(exc).lower()
                rate_limited = status == 429 or (status == 413 and "tokens per minute" in text) or " 429" in text
                if not rate_limited:
                    saw_timeout = isinstance(exc, TimeoutError) or "timeout" in text
                    error = str(exc)[:300]
                    raise
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", {}) or {}
                _HEADER_RATE_TRACKER.update(str(active_model), headers)
                if quick_mode and "tokens per day (tpd)" in text:
                    reset_match = re.search(r"try again in\s+(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?", text)
                    reset_seconds = (float(reset_match.group(1) or 0) * 60 + float(reset_match.group(2) or 0)) if reset_match else 900.0
                    _HEADER_RATE_TRACKER.mark_unavailable(str(active_model), reset_seconds)
                saw_429 = saw_429 or status == 429 or "429" in text
                if status == 429 or "429" in text:
                    rate_limit_count += 1
                else:
                    tpm_limit_count += 1
                retry_header = headers.get("retry-after") or headers.get("Retry-After") or headers.get("x-ratelimit-reset-tokens")
                match = re.match(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m)?", str(retry_header), re.I) if retry_header else None
                amount = float(match.group(1)) if match else 1.5 * (2 ** retries)
                unit = match.group(2).lower() if match and match.group(2) else "s"
                delay = amount / 1000 if unit == "ms" else amount * 60 if unit == "m" else amount
                delay = min(max(0.0, delay), 15.0 if quick_mode else max(0.0, 20 - retry_after_seconds))
                if quick_mode:
                    if delay:
                        time.sleep(delay)
                        retry_after_seconds += delay
                        if telemetry:
                            telemetry.record_sleep(delay, "rate_limit_backoff", str(active_model), call_id)
                    break
                if retries >= 2 or retry_after_seconds >= 20:
                    break
                delay = min(max(0.1, delay), max(0.0, 20 - retry_after_seconds))
                retries += 1
                time.sleep(delay)
                retry_after_seconds += delay
                if telemetry:
                    telemetry.record_sleep(delay, "rate_limit_backoff", str(active_model), call_id)

        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise MissingAPIKeyError("GROQ_API_KEY is missing. Add it to your .env file.")
        if quick_mode:
            candidates = [QUICK_FALLBACK_MODEL, FALLBACK_MODEL, DEFAULT_MODEL]
            fallback_model = next((item for item in candidates if item != active_model and not _HEADER_RATE_TRACKER.is_unavailable(item)), next(item for item in candidates if item != active_model))
        else:
            fallback_model = DEFAULT_MODEL if active_model == FALLBACK_MODEL else FALLBACK_MODEL
        fallback_tokens = min(completion_budget, 700) if quick_mode else completion_budget
        fallback = make_llm(fallback_model, max_tokens=fallback_tokens)
        if not quick_mode:
            _TPM_BUDGET.reserve(estimate)
        try:
            active_model = fallback_model
            if quick_mode and _HEADER_RATE_TRACKER.is_unavailable(str(active_model)):
                raise RuntimeError(f"Fallback model {active_model} is currently blocked by its daily token quota.")
            if quick_mode:
                with _HEADER_RATE_TRACKER.model_lock(str(active_model)):
                    waited = _HEADER_RATE_TRACKER.wait_if_needed(str(active_model), estimate)
                    if telemetry and waited:
                        telemetry.record_sleep(waited, "header_token_wait", str(active_model), call_id)
                attempt_count += 1
                result = fallback.invoke(messages, **invoke_options)
            else:
                attempt_count += 1
                result = fallback.invoke(messages, **invoke_options)
            finish = (getattr(result, "response_metadata", {}) or {}).get("finish_reason", "")
            if detect_truncation and finish == "length":
                raise TruncatedOutputError("LLM response ended at the token limit (finish_reason=length).")
            return result
        except Exception as exc:
            saw_timeout = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
            fallback_status = getattr(exc, "status_code", None)
            if fallback_status == 429 or "429" in str(exc):
                saw_429 = True
                rate_limit_count += 1
            elif fallback_status == 413 and "tokens per minute" in str(exc).lower():
                tpm_limit_count += 1
            error = str(exc)[:300]
            raise
    except Exception as exc:
        saw_timeout = saw_timeout or isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
        error = error or str(exc)[:300]
        raise
    finally:
        if telemetry:
            telemetry.record({"kind": "llm", "name": "chat_completion", "call_id": call_id, "duration_ms": round((time.monotonic() - started) * 1000, 1), "model": str(active_model), "requested_model": str(getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"), "prompt_chars": prompt_chars, "approx_tokens": math.ceil(prompt_chars / 4), "retry_count": retries, "attempt_count": attempt_count, "primary_skipped": primary_skipped, "rate_limited": saw_429, "rate_limit_count": rate_limit_count, "tpm_limit_count": tpm_limit_count, "timeout": saw_timeout, "error": error, "sleep_seconds": round(retry_after_seconds, 3), "fallback_used": str(active_model) != str(getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown")})


class TruncatedOutputError(RuntimeError):
    """Raised when a model signals it stopped at the output token limit."""


def content_text(response: Any) -> str:
    """Normalize a chat response's content to text."""
    content = response.content
    return content if isinstance(content, str) else "\n".join(str(part) for part in content)


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a fenced JSON response."""
    import json

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    result = json.loads(cleaned[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def normalize_citations(report: str, sources: list[dict[str, Any]]) -> str:
    """Normalize inline numeric citations and build the linked source list."""
    import re

    source_heading = re.search(r"(?im)^#{1,6}\s+Sources\s*$", report)
    split_at = source_heading.start() if source_heading else len(report)
    body = report[:split_at].strip()
    valid = {int(item["id"]): str(item.get("url", "")) for item in sources if str(item.get("url", "")).startswith(("http://", "https://"))}

    def link_id(match: re.Match[str]) -> str:
        source_id = int(match.group(1))
        url = valid.get(source_id)
        return f"[{source_id}](<{url}>)" if url else ""

    body = re.sub(r"【(\d+)†[^】]*】", link_id, body)
    body = re.sub(r"(?<!\w)\[(\d+)\](?!\()", link_id, body)

    source_lines: list[str] = []
    for source in sources:
        url = str(source.get("url", "")).strip()
        title = str(source.get("title", "")).strip() or "Source"
        if not url.startswith(("http://", "https://")) or not (source.get("text") or source.get("snippet")):
            continue
        source_lines.append(f"[{source['id']}] [{title}](<{url}>)")
    catalog = "\n".join(source_lines) if source_lines else "No readable sources were retrieved."
    return f"{body}\n\n## Sources\n{catalog}\n"
