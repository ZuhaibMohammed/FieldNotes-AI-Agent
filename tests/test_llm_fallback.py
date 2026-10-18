"""Regression tests for model fallback after rate limits."""

from types import SimpleNamespace

import agent.llm as llm_module


class RateLimited(Exception):
    """Mimic an HTTP 429 response from Groq."""

    status_code = 429


def test_rate_limited_20b_primary_tries_120b(monkeypatch) -> None:
    """A 20B primary can still fall back to the 120B model."""
    primary = SimpleNamespace(model_name=llm_module.FALLBACK_MODEL, max_tokens=32)
    primary.invoke = lambda *_args, **_kwargs: (_ for _ in ()).throw(RateLimited("429 rate limited"))
    alternate = SimpleNamespace(
        model_name=llm_module.DEFAULT_MODEL,
        max_tokens=32,
        invoke=lambda *_args, **_kwargs: SimpleNamespace(content="ok", response_metadata={}),
    )

    monkeypatch.setattr(llm_module, "make_llm", lambda *_args, **_kwargs: alternate)
    monkeypatch.setattr(llm_module.time, "sleep", lambda *_args, **_kwargs: None)

    result = llm_module.invoke_with_fallback(primary, [])

    assert result.content == "ok"
