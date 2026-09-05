"""JSON Lines manifest persistence with strict validation and atomic replacement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time

from core.errors import CodeNovaError

LOGGER = logging.getLogger(__name__)

# On Windows, os.replace() can transiently fail with PermissionError/WinError 5
# when another process (antivirus, search indexer) briefly holds a handle on a
# just-written file — not a real conflict, just needs a moment to clear.
_REPLACE_MAX_RETRIES = 5
_REPLACE_RETRY_SECONDS = 0.05


class ManifestError(CodeNovaError):
    """Raised when a manifest is corrupt or violates an invariant."""


@dataclass(frozen=True)
class CorruptManifestLine:
    line_number: int
    content: str
    error: str


@dataclass(frozen=True)
class ManifestReadResult:
    rows: list[dict[str, object]]
    corrupt_lines: list[CorruptManifestLine]

    @property
    def valid(self) -> bool:
        return not self.corrupt_lines


class JsonlManifest:
    """Append, inspect and atomically replace JSONL records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(self._serialize(record) + "\n")
            handle.flush()

    def extend(self, records: Iterable[dict[str, object]]) -> None:
        serialized = [self._serialize(record) for record in records]
        if not serialized:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(serialized) + "\n")
            handle.flush()

    def inspect(self) -> ManifestReadResult:
        if not self.path.exists():
            return ManifestReadResult([], [])
        rows: list[dict[str, object]] = []
        corrupt: list[CorruptManifestLine] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                    if not isinstance(value, dict):
                        raise ValueError("JSONL record must be an object")
                    rows.append(value)
                except (json.JSONDecodeError, ValueError) as exc:
                    corrupt.append(CorruptManifestLine(line_number, stripped[:500], str(exc)))
        return ManifestReadResult(rows, corrupt)

    def read_all(self, *, strict: bool = False) -> list[dict[str, object]]:
        result = self.inspect()
        if strict and result.corrupt_lines:
            first = result.corrupt_lines[0]
            raise ManifestError(f"{self.path}:{first.line_number}: {first.error}")
        for line in result.corrupt_lines:
            LOGGER.warning(
                "Skipping corrupt line %s in %s: %s", line.line_number, self.path, line.error
            )
        return result.rows

    def ids(self, key: str, *, strict: bool = False) -> set[str]:
        return {str(row[key]) for row in self.read_all(strict=strict) if key in row}

    def validate_unique(self, key: str) -> None:
        rows = self.read_all(strict=True)
        missing = [index for index, row in enumerate(rows) if key not in row]
        if missing:
            raise ManifestError(f"{self.path}: records missing {key!r}: {missing[:20]}")
        counts = Counter(str(row[key]) for row in rows)
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        if duplicates:
            raise ManifestError(f"{self.path}: duplicate {key!r}: {duplicates[:20]}")

    def replace_all(
        self,
        records: Iterable[dict[str, object]],
        *,
        unique_key: str | None = None,
        backup: bool = False,
    ) -> None:
        rows = list(records)
        self._validate_rows(rows, unique_key)
        self._atomic_write(rows, backup=backup)

    def repair_corrupt_lines(
        self, *, dry_run: bool = True, backup: bool = True
    ) -> ManifestReadResult:
        result = self.inspect()
        if result.corrupt_lines and not dry_run:
            self._atomic_write(result.rows, backup=backup)
        return result

    @staticmethod
    def _serialize(record: dict[str, object]) -> str:
        if not isinstance(record, dict):
            raise ManifestError("Manifest record must be a dictionary")
        return json.dumps(record, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _validate_rows(rows: list[dict[str, object]], unique_key: str | None) -> None:
        if unique_key is None:
            return
        missing = [index for index, row in enumerate(rows) if unique_key not in row]
        if missing:
            raise ManifestError(f"Records missing {unique_key!r}: {missing[:20]}")
        counts = Counter(str(row[unique_key]) for row in rows)
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        if duplicates:
            raise ManifestError(f"Duplicate {unique_key!r}: {duplicates[:20]}")

    def _atomic_write(self, rows: list[dict[str, object]], *, backup: bool) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(self._serialize(row) + "\n")
            handle.flush()
        try:
            parsed = JsonlManifest(temporary).read_all(strict=True)
            if len(parsed) != len(rows):
                raise ManifestError(
                    f"Temporary manifest count mismatch expected={len(rows)} actual={len(parsed)}"
                )
            if backup and self.path.exists():
                backup_path = self.path.with_name(
                    f"{self.path.name}.{self.path.stat().st_mtime_ns}.bak"
                )
                shutil.copy2(self.path, backup_path)
            self._replace_with_retry(temporary)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _replace_with_retry(self, temporary: Path) -> None:
        for attempt in range(_REPLACE_MAX_RETRIES):
            try:
                temporary.replace(self.path)
                return
            except PermissionError:
                if attempt == _REPLACE_MAX_RETRIES - 1:
                    raise
                time.sleep(_REPLACE_RETRY_SECONDS * (attempt + 1))
