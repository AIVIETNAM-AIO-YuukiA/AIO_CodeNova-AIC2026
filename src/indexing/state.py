"""SQLite-backed pipeline state for resumable long-running jobs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
import sqlite3

_VALID_STATUSES = {
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "COMPLETED_NO_OUTPUT",
    "BLOCKED",
    "FAILED",
}

# sqlite3's default connect() timeout is 5s: with CAPTION_WORKERS threads each
# opening their own connection and writing after every OCR call, concurrent
# writers can queue up past that under load, raising "database is locked"
# instead of just waiting a bit longer for the single writer to finish.
_CONNECT_TIMEOUT_SECONDS = 30.0


class JobState:
    """Persist per-item status for pipeline stages."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection that commits on success."""
        connection = sqlite3.connect(self.path, timeout=_CONNECT_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        # WAL lets readers (get_status/completed_ids) proceed without blocking
        # on the single writer (mark), instead of the default rollback-journal
        # mode where any write briefly locks out all readers too — the main
        # source of "database is locked" under CAPTION_WORKERS concurrency.
        connection.execute("PRAGMA journal_mode=WAL")
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
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unknown job status {status!r}")
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
        return not force and self.get_status(item_id, stage) in {
            "COMPLETED",
            "COMPLETED_NO_OUTPUT",
        }

    def completed_ids(self, stage: str) -> set[str]:
        """Return every item_id already COMPLETED/COMPLETED_NO_OUTPUT for a stage.

        One query for bulk skip-filtering instead of ``should_skip`` per item,
        which opens a fresh sqlite connection per call — measured at minutes
        of pure connection overhead over hundreds of thousands of frames.
        """
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT item_id FROM jobs WHERE stage = ? "
                "AND status IN ('COMPLETED', 'COMPLETED_NO_OUTPUT')",
                (stage,),
            ).fetchall()
        return {str(row["item_id"]) for row in rows}

    def failures(self, stage: str | None = None) -> list[dict[str, object]]:
        """Return current failed/blocked items for validation and retry planning."""
        query = (
            "SELECT item_id, stage, status, updated_at, attempts, error "
            "FROM jobs WHERE status IN ('FAILED', 'BLOCKED')"
        )
        params: tuple[object, ...] = ()
        if stage is not None:
            query += " AND stage = ?"
            params = (stage,)
        query += " ORDER BY updated_at"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]
