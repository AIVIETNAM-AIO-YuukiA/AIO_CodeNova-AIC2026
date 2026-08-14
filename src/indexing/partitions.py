"""Atomic per-item partitions consolidated into legacy-compatible manifests."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from indexing.manifest import JsonlManifest, ManifestError


class PartitionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, partition_id: str) -> Path:
        if (
            not partition_id
            or partition_id in {".", ".."}
            or "/" in partition_id
            or "\\" in partition_id
        ):
            raise ManifestError(f"Unsafe partition ID: {partition_id!r}")
        return self.root / f"{partition_id}.jsonl"

    def exists(self, partition_id: str) -> bool:
        return self.path_for(partition_id).exists()

    def write(
        self,
        partition_id: str,
        records: Iterable[dict[str, object]],
        *,
        partition_key: str | None = None,
        unique_key: str | None = None,
    ) -> Path:
        rows = list(records)
        if partition_key and any(str(row.get(partition_key)) != partition_id for row in rows):
            raise ManifestError(
                f"Partition {partition_id!r} contains records with another {partition_key}"
            )
        destination = self.path_for(partition_id)
        JsonlManifest(destination).replace_all(rows, unique_key=unique_key)
        return destination

    def consolidate(self, destination: Path, *, unique_key: str) -> Path:
        rows: list[dict[str, object]] = []
        for path in sorted(self.root.glob("*.jsonl")):
            rows.extend(JsonlManifest(path).read_all(strict=True))
        JsonlManifest(destination).replace_all(rows, unique_key=unique_key, backup=True)
        return destination
