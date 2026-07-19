"""SQLite-backed pipeline state for resumable long-running jobs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
import sqlite3


class JobState:
    """Persist per-item status for pipeline stages."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection that commits on success."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    item_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    PRIMARY KEY (item_id, stage)
                )
                """
            )

    def get_status(self, item_id: str, stage: str) -> str | None:
        """Return the current status for an item/stage pair."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status FROM jobs WHERE item_id = ? AND stage = ?",
                (item_id, stage),
            ).fetchone()
        return None if row is None else str(row["status"])

    def mark(self, item_id: str, stage: str, status: str, error: str | None = None) -> None:
        """Set status for an item/stage pair and increment attempts on failure."""
        updated_at = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (item_id, stage, status, updated_at, attempts, error)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id, stage) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    attempts = jobs.attempts + CASE
                        WHEN excluded.status = 'FAILED' THEN 1 ELSE 0
                    END,
                    error = excluded.error
                """,
                (item_id, stage, status, updated_at, 1 if status == "FAILED" else 0, error),
            )

    def should_skip(self, item_id: str, stage: str, force: bool = False) -> bool:
        """Return whether a completed item should be skipped."""
        return not force and self.get_status(item_id, stage) == "COMPLETED"
