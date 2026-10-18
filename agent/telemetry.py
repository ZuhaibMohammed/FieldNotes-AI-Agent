"""Per-run timing, retry, and timeout telemetry for the research workflow."""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


_ACTIVE: contextvars.ContextVar[RunTelemetry | None] = contextvars.ContextVar("research_telemetry", default=None)
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "research_agent.jsonl"


class RunTelemetry:
    """Collect operations for one run and append safe JSONL diagnostics."""

    def __init__(self, mode: str, time_budget: float) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.started_at = time.monotonic()
        self.deadline = self.started_at + time_budget
        self.time_budget = time_budget
        self.events: list[dict[str, Any]] = []
        self.sleep_seconds = 0.0
        self._lock = threading.Lock()

    def remaining(self) -> float:
        """Return the number of seconds left in this run's budget."""
        return max(0.0, self.deadline - time.monotonic())

    def expired(self) -> bool:
        """Return whether the run has exhausted its time budget."""
        return self.remaining() <= 0

    def record_sleep(self, seconds: float, reason: str, model: str = "", call_id: str = "") -> None:
        """Track time spent waiting for rate limits or token budget."""
        item = {"run_id": self.run_id, "mode": self.mode, "kind": "sleep", "name": reason, "duration_ms": round(seconds * 1000, 1), "model": model, "call_id": call_id, "prompt_chars": 0, "approx_tokens": 0, "retry_count": 0, "rate_limited": False, "timeout": False, "error": "", "sleep_seconds": round(seconds, 3)}
        with self._lock:
            self.sleep_seconds += max(0.0, seconds)
            self.events.append(item)
        self._flush(item)

    def record(self, event: dict[str, Any]) -> None:
        """Save a structured timing event to memory and the JSONL log."""
        item = {"run_id": self.run_id, "mode": self.mode, "duration_ms": 0.0, "model": "", "prompt_chars": 0, "approx_tokens": 0, "retry_count": 0, "rate_limited": False, "timeout": False, "error": "", **event}
        with self._lock:
            self.events.append(item)
        self._flush(item)

    def _flush(self, event: dict[str, Any]) -> None:
        """Append one JSON record without storing prompt contents."""
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as log:
                log.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def node_table(self) -> list[dict[str, Any]]:
        """Aggregate node timings for display after a run."""
        groups: dict[str, list[float]] = defaultdict(list)
        for event in self.events:
            if event.get("kind") == "node":
                groups[str(event["name"])].append(float(event.get("duration_ms", 0)))
        return [{"Node": name, "Calls": len(times), "Total (s)": round(sum(times) / 1000, 2), "Mean (s)": round(sum(times) / len(times) / 1000, 2)} for name, times in sorted(groups.items())]

    def counts(self) -> dict[str, Any]:
        """Summarize model, network, retry, and wait totals."""
        llm_events = [event for event in self.events if event.get("kind") == "llm"]
        return {
            "llm_calls": sum(int(event.get("attempt_count", 1 + int(event.get("retry_count", 0)) + int(bool(event.get("fallback_used"))))) for event in llm_events),
            "llm_invocations": len(llm_events),
            "fetches": sum(event.get("kind") == "fetch" for event in self.events),
            "searches": sum(event.get("kind") == "search" for event in self.events),
            "429s": sum(int(event.get("rate_limit_count", bool(event.get("rate_limited")))) for event in self.events if event.get("kind") in {"llm", "search", "fetch"}),
            "TPM limit hits": sum(int(event.get("tpm_limit_count", 0)) for event in self.events if event.get("kind") == "llm"),
            "fallbacks": sum(bool(event.get("fallback_used")) for event in self.events if event.get("kind") == "llm"),
            "warnings": sum(event.get("kind") == "warning" for event in self.events),
            "timeouts": sum(event.get("kind") in {"llm", "search", "fetch"} and bool(event.get("timeout")) for event in self.events),
            "total_seconds": round(time.monotonic() - self.started_at, 2),
            "sleep_seconds": round(self.sleep_seconds, 2),
        }


def current_telemetry() -> RunTelemetry | None:
    """Get the telemetry context active in this execution context."""
    return _ACTIVE.get()


def activate_telemetry(telemetry: RunTelemetry) -> contextvars.Token[RunTelemetry | None]:
    """Bind one run's telemetry to the current thread or async task."""
    return _ACTIVE.set(telemetry)


def reset_telemetry(token: contextvars.Token[RunTelemetry | None]) -> None:
    """Restore the prior telemetry context."""
    _ACTIVE.reset(token)


def instrument_node(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a graph node and record its duration and failure status."""
    import inspect
    from functools import wraps

    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def async_wrapped(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            telemetry = current_telemetry()
            started = time.monotonic()
            event_offset = len(telemetry.events) if telemetry else 0
            error: BaseException | None = None
            try:
                return await function(state, *args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                if telemetry:
                    calls = [item for item in telemetry.events[event_offset:] if item.get("kind") == "llm"]
                    telemetry.record({"kind": "node", "name": name, "duration_ms": round((time.monotonic() - started) * 1000, 1), "model": ", ".join(dict.fromkeys(str(item.get("model", "")) for item in calls if item.get("model"))) or str(state.get("model", "")), "prompt_chars": sum(int(item.get("prompt_chars", 0)) for item in calls), "approx_tokens": sum(int(item.get("approx_tokens", 0)) for item in calls), "retry_count": sum(int(item.get("retry_count", 0)) for item in calls), "rate_limited": _is_429(error) or any(item.get("rate_limited") for item in calls), "timeout": _is_timeout(error) or any(item.get("timeout") for item in calls), "error": str(error)[:300] if error else ""})
        return async_wrapped

    @wraps(function)
    def wrapped(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        telemetry = current_telemetry()
        started = time.monotonic()
        event_offset = len(telemetry.events) if telemetry else 0
        error: BaseException | None = None
        try:
            return function(state, *args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            if telemetry:
                calls = [item for item in telemetry.events[event_offset:] if item.get("kind") == "llm"]
                telemetry.record({"kind": "node", "name": name, "duration_ms": round((time.monotonic() - started) * 1000, 1), "model": ", ".join(dict.fromkeys(str(item.get("model", "")) for item in calls if item.get("model"))) or str(state.get("model", "")), "prompt_chars": sum(int(item.get("prompt_chars", 0)) for item in calls), "approx_tokens": sum(int(item.get("approx_tokens", 0)) for item in calls), "retry_count": sum(int(item.get("retry_count", 0)) for item in calls), "rate_limited": _is_429(error) or any(item.get("rate_limited") for item in calls), "timeout": _is_timeout(error) or any(item.get("timeout") for item in calls), "error": str(error)[:300] if error else ""})
    return wrapped


def _is_429(error: BaseException | None) -> bool:
    """Check an exception for a rate-limit response."""
    return bool(error and (getattr(error, "status_code", None) == 429 or "429" in str(error)))


def _is_timeout(error: BaseException | None) -> bool:
    """Check an exception for a timeout condition."""
    return bool(error and (isinstance(error, TimeoutError) or "timeout" in str(error).lower()))
