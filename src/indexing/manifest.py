"""JSON Lines manifest persistence."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from pathlib import Path
import json

LOGGER = logging.getLogger(__name__)


class JsonlManifest:
    """Append/read JSONL records for resumable pipeline outputs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, object]) -> None:
        """Append one JSON-serializable record."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def extend(self, records: Iterable[dict[str, object]]) -> None:
        """Append several records."""
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, object]]:
        """Read all records from the manifest.

        Skips (with a warning) any line that fails to parse instead of
        raising — a process killed mid-write can leave a truncated/garbage
        last line (e.g. null bytes from an unflushed disk block), and losing
        the whole manifest over one bad trailing line would throw away every
        record that came before it.
        """
        if not self.path.exists():
            return []
        rows: list[dict[str, object]] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    LOGGER.warning(
                        "Skipping corrupt line %s in %s (likely a killed mid-write)",
                        line_number,
                        self.path,
                    )
        return rows

    def ids(self, key: str) -> set[str]:
        """Return string IDs already recorded under ``key``."""
        return {str(row[key]) for row in self.read_all() if key in row}
