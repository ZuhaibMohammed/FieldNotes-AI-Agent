"""Pluggable web search and page extraction tools."""

from __future__ import annotations

import os
import time
import asyncio
import math
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from dotenv import load_dotenv

from agent.telemetry import current_telemetry


def normalize_results(results: list[dict[str, Any]], max_results: int) -> list[dict[str, str]]:
    """Normalize provider results to title, url and snippet and deduplicate URLs."""
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in results:
        url = str(result.get("url") or result.get("href") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        normalized.append({
            "title": str(result.get("title") or "Untitled").strip(),
            "url": url,
            "snippet": str(result.get("snippet") or result.get("body") or result.get("content") or "").strip(),
        })
        if len(normalized) >= max_results:
            break
    return normalized


def search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search once with Tavily when configured, otherwise DuckDuckGo."""
    load_dotenv()
    telemetry = current_telemetry()
    started = time.monotonic()
    retry_count = 0
    timed_out = False
    error = ""
    results: list[dict[str, str]] = []
    try:
        if os.getenv("TAVILY_API_KEY"):
            from tavily import TavilyClient
            response = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query=query, max_results=max_results, timeout=8)
            results = normalize_results(response.get("results", []), max_results)
        else:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = normalize_results(list(DDGS(timeout=8).text(query, max_results=max_results)), max_results)
        return results
    except (TimeoutError, httpx.TimeoutException) as exc:
        timed_out = True
        error = str(exc)[:200]
        raise
    except Exception as exc:
        error = str(exc)[:200]
        raise RuntimeError(f"Web search failed: {exc}") from exc
    finally:
        if telemetry:
            telemetry.record({"kind": "search", "name": query, "duration_ms": round((time.monotonic() - started) * 1000, 1), "model": "", "prompt_chars": len(query), "approx_tokens": math.ceil(len(query) / 4), "retry_count": retry_count, "rate_limited": "429" in error, "rate_limit_count": int("429" in error), "timeout": timed_out, "error": error, "result_count": len(results)})


async def fetch_pages(urls: list[str], timeout: float = 6.0, max_chars: int = 3000) -> dict[str, str]:
    """Fetch URLs concurrently with a hard timeout and no retry."""
    telemetry = current_telemetry()
    timeout = max(0.05, min(6.0, timeout))
    limits = httpx.Limits(max_connections=max(1, len(urls)), max_keepalive_connections=max(1, len(urls)))
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True, limits=limits, headers={"User-Agent": "ResearchAgent/1.0"}) as client:
        async def fetch_one(url: str) -> tuple[str, str]:
            started = time.monotonic()
            error = ""
            timed_out = False
            result = ""
            try:
                async with asyncio.timeout(timeout):
                    response = await client.get(url)
                    response.raise_for_status()
                    extracted = await asyncio.to_thread(trafilatura.extract, response.text, include_comments=False, include_tables=False)
                    result = (extracted or "").strip()[:max_chars]
            except (TimeoutError, httpx.TimeoutException) as exc:
                timed_out = True
                error = str(exc) or "Fetch timed out."
            except Exception as exc:
                error = str(exc)[:200]
            finally:
                if telemetry:
                    telemetry.record({"kind": "fetch", "name": url, "duration_ms": round((time.monotonic() - started) * 1000, 1), "model": "", "prompt_chars": 0, "approx_tokens": 0, "retry_count": 0, "rate_limited": False, "timeout": timed_out, "error": error, "chars_extracted": len(result)})
            return url, result

        pairs = await asyncio.gather(*(fetch_one(url) for url in dict.fromkeys(urls)), return_exceptions=True)
    return {url: content for pair in pairs if isinstance(pair, tuple) for url, content in [pair]}


def fetch_page(url: str, timeout: float = 6.0, max_chars: int = 3000) -> str:
    """Fetch one page through the async implementation for CLI compatibility."""
    return asyncio.run(fetch_pages([url], timeout=timeout, max_chars=max_chars)).get(url, "")


def domain_name(url: str) -> str:
    """Return a human-readable host name for a URL."""
    return urlparse(url).netloc.removeprefix("www.")
