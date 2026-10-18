"""SQLite persistence for completed research reports."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "research_history.sqlite3"


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open the history database and ensure its table exists."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS research_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, created_at TEXT NOT NULL, report TEXT NOT NULL, sources_json TEXT NOT NULL)")
    db.commit()
    return db


@contextmanager
def _session(path: str | Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Commit a database operation and always close its SQLite connection."""
    db = connect(path)
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def save_run(topic: str, report: str, sources: list[dict[str, Any]], path: str | Path = DB_PATH) -> int:
    """Persist a completed report and its source catalog."""
    with _session(path) as db:
        cur = db.execute("INSERT INTO research_runs(topic, created_at, report, sources_json) VALUES (?, ?, ?, ?)", (topic, datetime.now(timezone.utc).isoformat(timespec="seconds"), report, json.dumps(sources, ensure_ascii=False)))
        return int(cur.lastrowid)


def list_runs(path: str | Path = DB_PATH, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent history entries for the sidebar."""
    with _session(path) as db:
        rows = db.execute("SELECT id, topic, created_at, report FROM research_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": row[0], "topic": row[1], "created_at": row[2], "snippet": _report_snippet(row[3])} for row in rows]


def _report_snippet(report: str) -> str:
    """Extract a short plain-text preview for the history library."""
    text = report.replace("\r", " ").replace("\n", " ")
    text = text.replace("#", "").replace("|", " ").replace("[", "").replace("]", "")
    return " ".join(text.split())[:180]


def get_run(run_id: int, path: str | Path = DB_PATH) -> dict[str, Any] | None:
    """Load a saved report by its id."""
    with _session(path) as db:
        row = db.execute("SELECT id, topic, created_at, report, sources_json FROM research_runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        return None
    return {"id": row[0], "topic": row[1], "created_at": row[2], "report": row[3], "sources": json.loads(row[4])}


def delete_run(run_id: int, path: str | Path = DB_PATH) -> bool:
    """Delete one saved report by id and report whether a row was removed."""
    with _session(path) as db:
        cursor = db.execute("DELETE FROM research_runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0
