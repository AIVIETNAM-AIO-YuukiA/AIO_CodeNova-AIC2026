"""Inspect and migrate legacy frame paths into experiment-owned canonical paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

from config.settings import Experiment
from core.paths import canonical_frame_path
from indexing.manifest import JsonlManifest, ManifestError


@dataclass(frozen=True)
class FramePathMigrationIssue:
    manifest: str
    frame_id: str
    raw_path: str | None
    reason: str


@dataclass
class FramePathMigrationPlan:
    experiment: str
    legacy_root: str
    manifests: dict[Path, list[dict[str, object]]]
    changed_records: int
    issues: list[FramePathMigrationIssue]

    def to_dict(self, *, mode: str, changed: bool = False) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "mode": mode,
            "legacy_root": self.legacy_root,
            "manifests_checked": [str(path) for path in self.manifests],
            "changed_records": self.changed_records,
            "issues": [asdict(issue) for issue in self.issues],
            "changed": changed,
        }


def plan_frame_path_migration(experiment: Experiment, legacy_root: Path) -> FramePathMigrationPlan:
    """Build a read-only migration plan for the aggregate and partition manifests."""
    manifests_root = experiment.run_dir / "manifests"
    paths = [manifests_root / "frames.jsonl"]
    partition_root = manifests_root / "partitions" / "frames"
    paths.extend(sorted(partition_root.glob("*.jsonl")))

    migrated: dict[Path, list[dict[str, object]]] = {}
    issues: list[FramePathMigrationIssue] = []
    changed_records = 0
    for path in paths:
        if not path.exists():
            continue
        rows = JsonlManifest(path).read_all(strict=True)
        migrated_rows: list[dict[str, object]] = []
        for row in rows:
            updated = dict(row)
            raw_value = row.get("frame_path")
            raw = str(raw_value).replace("\\", "/") if raw_value else None
            resolved = _resolve_legacy_path(experiment, legacy_root, raw)
            if resolved is None:
                issues.append(
                    FramePathMigrationIssue(
                        str(path),
                        str(row.get("frame_id", "")),
                        raw,
                        "FRAME_FILE_MISSING_OR_OUTSIDE_EXPERIMENT",
                    )
                )
            else:
                canonical = canonical_frame_path(experiment, resolved)
                if canonical != raw:
                    updated["frame_path"] = canonical
                    changed_records += 1
            migrated_rows.append(updated)
        migrated[path] = migrated_rows
    return FramePathMigrationPlan(
        experiment.name,
        str(legacy_root.resolve()),
        migrated,
        changed_records,
        issues,
    )


def apply_frame_path_migration(experiment: Experiment, plan: FramePathMigrationPlan) -> Path:
    """Apply a completely valid plan atomically per manifest and invalidate readiness."""
    if plan.issues:
        raise ManifestError(
            f"Refusing frame-path migration with {len(plan.issues)} unresolved record(s)"
        )
    for path, rows in plan.manifests.items():
        JsonlManifest(path).replace_all(rows, unique_key="frame_id", backup=True)
    (experiment.run_dir / "readiness.json").unlink(missing_ok=True)
    return write_frame_path_migration_audit(experiment, plan, mode="APPLY", changed=True)


def write_frame_path_migration_audit(
    experiment: Experiment,
    plan: FramePathMigrationPlan,
    *,
    mode: str,
    changed: bool = False,
) -> Path:
    audit_dir = experiment.run_dir / "logs" / "migrations"
    audit_dir.mkdir(parents=True, exist_ok=True)
    destination = audit_dir / f"frame_paths_{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    payload = {
        **plan.to_dict(mode=mode, changed=changed),
        "created_at": datetime.now(UTC).isoformat(),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _resolve_legacy_path(experiment: Experiment, legacy_root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    candidates = (
        [path] if path.is_absolute() else [experiment.run_dir / path, legacy_root.resolve() / path]
    )
    run_root = experiment.run_dir.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(run_root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None
