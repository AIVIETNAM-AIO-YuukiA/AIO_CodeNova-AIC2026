"""JSON Lines manifest persistence."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json


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
        """Read all records from the manifest."""
        if not self.path.exists():
            return []
        rows: list[dict[str, object]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    def ids(self, key: str) -> set[str]:
        """Return string IDs already recorded under ``key``."""
        return {str(row[key]) for row in self.read_all() if key in row}
