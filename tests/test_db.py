"""Tests for SQLite research history operations."""

from pathlib import Path

from agent.db import delete_run, get_run, save_run


def test_delete_run_removes_only_the_requested_report(tmp_path: Path) -> None:
    """Deleting one report leaves other history entries intact."""
    database = tmp_path / "history.sqlite3"
    first_id = save_run("First report", "# First", [], database)
    second_id = save_run("Second report", "# Second", [], database)

    assert delete_run(first_id, database)
    assert get_run(first_id, database) is None
    assert get_run(second_id, database)["topic"] == "Second report"
    assert not delete_run(first_id, database)
