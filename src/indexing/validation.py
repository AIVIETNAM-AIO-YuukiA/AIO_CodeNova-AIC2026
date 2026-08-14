"""Cross-stage quality gate for offline indexing artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from config.settings import Experiment
from core.paths import resolve_experiment_frame_path
from indexing.embedding_paths import frame_ids_path, vectors_path
from indexing.manifest import JsonlManifest
from indexing.state import JobState
from modules.embedding import resolve_embedding_model
from core.errors import EmbeddingError


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    stage: str
    message: str
    item_id: str | None = None
    artifact: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ExperimentValidationReport:
    experiment: str
    status: str = "READY"
    issues: list[ValidationIssue] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, dict[str, object]] = field(default_factory=dict)

    def add(
        self,
        severity: str,
        code: str,
        stage: str,
        message: str,
        *,
        item_id: str | None = None,
        artifact: Path | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity, code, stage, message, item_id, str(artifact) if artifact else None
            )
        )
        if severity == "ERROR":
            self.status = "INVALID"
        elif self.status == "READY":
            self.status = "DEGRADED"

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
            "coverage": self.coverage,
            "artifacts": self.artifacts,
        }


def validate_experiment_artifacts(experiment: Experiment) -> ExperimentValidationReport:
    """Validate manifests, references, files and every configured embedding artifact."""
    report = ExperimentValidationReport(experiment.name)
    manifests = experiment.run_dir / "manifests"
    videos = _read(report, manifests / "videos.jsonl", "DISCOVER", required=True)
    shots = _read(report, manifests / "shots.jsonl", "SHOT_DETECT", required=True)
    frames = _read(report, manifests / "frames.jsonl", "FRAME_EXTRACT", required=True)
    text = _read(report, manifests / "text.jsonl", "TEXT", required=False)
    captions = _read(report, manifests / "captions.jsonl", "CAPTION", required=False)
    embedding_records = _read(report, manifests / "embeddings.jsonl", "EMBED", required=True)

    video_ids = _unique(report, videos, "video_id", "DISCOVER")
    shot_ids = _unique(report, shots, "shot_id", "SHOT_DETECT")
    frame_ids = _unique(report, frames, "frame_id", "FRAME_EXTRACT")
    if text:
        _unique(report, text, "doc_id", "TEXT")
    if captions:
        _unique(report, captions, "frame_id", "CAPTION")
    if embedding_records:
        _unique(report, embedding_records, "model_name", "EMBED")

    missing_video_files = 0
    for video in videos:
        video_id = str(video.get("video_id", ""))
        raw_path = video.get("path")
        path = Path(str(raw_path)) if raw_path else None
        if path is None or not path.is_file():
            missing_video_files += 1
            report.add("ERROR", "MISSING_VIDEO_FILE", "DISCOVER", str(raw_path), item_id=video_id)
            continue
        try:
            expected_size = int(video["size_bytes"])
            expected_checksum = str(video["checksum"])
        except (KeyError, TypeError, ValueError) as exc:
            report.add("ERROR", "INVALID_VIDEO_METADATA", "DISCOVER", str(exc), item_id=video_id)
            continue
        if path.stat().st_size != expected_size:
            report.add(
                "ERROR",
                "VIDEO_SIZE_MISMATCH",
                "DISCOVER",
                f"expected={expected_size} actual={path.stat().st_size}",
                item_id=video_id,
            )
        else:
            try:
                source_fingerprint = _fingerprint(path)
            except OSError as exc:
                report.add("ERROR", "VIDEO_READ_FAILED", "DISCOVER", str(exc), item_id=video_id)
            else:
                report.artifacts[f"video-source:{video_id}"] = source_fingerprint
                if source_fingerprint["sha256"] != expected_checksum:
                    report.add(
                        "ERROR",
                        "VIDEO_CHECKSUM_MISMATCH",
                        "DISCOVER",
                        str(path),
                        item_id=video_id,
                    )

    shot_video: dict[str, str] = {}
    for shot in shots:
        shot_id = str(shot.get("shot_id", ""))
        video_id = str(shot.get("video_id", ""))
        shot_video[shot_id] = video_id
        if video_id not in video_ids:
            report.add("ERROR", "UNKNOWN_SHOT_VIDEO", "SHOT_DETECT", video_id, item_id=shot_id)
        try:
            if int(shot["end_frame"]) < int(shot["start_frame"]):
                raise ValueError("end_frame < start_frame")
        except (KeyError, TypeError, ValueError) as exc:
            report.add("ERROR", "INVALID_SHOT_BOUNDARY", "SHOT_DETECT", str(exc), item_id=shot_id)

    frame_path_counts = {
        "valid": 0,
        "missing_metadata": 0,
        "missing_file": 0,
        "outside_experiment": 0,
        "not_canonical": 0,
    }
    for frame in frames:
        frame_id = str(frame.get("frame_id", ""))
        video_id = str(frame.get("video_id", ""))
        shot_id = str(frame.get("shot_id", ""))
        if video_id not in video_ids:
            report.add("ERROR", "UNKNOWN_FRAME_VIDEO", "FRAME_EXTRACT", video_id, item_id=frame_id)
        if shot_id not in shot_ids:
            report.add("ERROR", "UNKNOWN_FRAME_SHOT", "FRAME_EXTRACT", shot_id, item_id=frame_id)
        elif shot_video.get(shot_id) != video_id:
            report.add(
                "ERROR",
                "FRAME_SHOT_VIDEO_MISMATCH",
                "FRAME_EXTRACT",
                f"frame video={video_id}, shot video={shot_video.get(shot_id)}",
                item_id=frame_id,
            )
        resolution = resolve_experiment_frame_path(experiment, frame.get("frame_path"))
        if not resolution.valid:
            counter = {
                "FRAME_PATH_MISSING": "missing_metadata",
                "FRAME_FILE_MISSING": "missing_file",
                "FRAME_PATH_OUTSIDE_EXPERIMENT": "outside_experiment",
            }.get(resolution.reason, "missing_file")
            frame_path_counts[counter] += 1
            report.add(
                "ERROR",
                resolution.reason or "FRAME_PATH_INVALID",
                "FRAME_EXTRACT",
                f"raw={resolution.raw_path!r} resolved={resolution.resolved_path}",
                item_id=frame_id,
            )
        elif not resolution.canonical:
            frame_path_counts["not_canonical"] += 1
            report.add(
                "ERROR",
                "FRAME_PATH_NOT_CANONICAL",
                "FRAME_EXTRACT",
                f"Use a path relative to experiment.run_dir: {resolution.raw_path!r}",
                item_id=frame_id,
            )
        else:
            frame_path_counts["valid"] += 1

    embedding_coverage: dict[str, object] = {}
    embedding_dir = experiment.run_dir / "embeddings"
    provenance_by_model = {
        str(row.get("model_name")): row for row in embedding_records if row.get("model_name")
    }
    configured_models = set(experiment.config.embedding_models)
    for extra_model in set(provenance_by_model) - configured_models:
        report.add(
            "ERROR",
            "EMBEDDING_MODEL_NOT_CONFIGURED",
            "EMBED",
            extra_model,
            item_id=extra_model,
        )
    for model in experiment.config.embedding_models:
        embedding_coverage[model] = _validate_embedding(
            report,
            embedding_dir,
            model,
            frame_ids,
            provenance_by_model.get(model),
            allow_partial=False,
        )
    embedding_alignment = _validate_cross_model_alignment(
        report, embedding_dir, experiment.config.embedding_models
    )

    for document in text:
        source = document.get("source")
        if source == "ocr" and str(document.get("frame_id") or "") not in frame_ids:
            report.add(
                "ERROR",
                "OCR_UNKNOWN_FRAME",
                "EXTRACT_OCR",
                str(document.get("frame_id")),
                item_id=str(document.get("doc_id", "")),
            )
        if source == "asr" and str(document.get("video_id") or "") not in video_ids:
            report.add(
                "ERROR",
                "ASR_UNKNOWN_VIDEO",
                "EXTRACT_ASR",
                str(document.get("video_id")),
                item_id=str(document.get("doc_id", "")),
            )
    for caption in captions:
        caption_frame = str(caption.get("frame_id") or "")
        if caption_frame not in frame_ids:
            report.add(
                "ERROR", "CAPTION_UNKNOWN_FRAME", "CAPTION", caption_frame, item_id=caption_frame
            )

    failures = JobState(experiment.run_dir / "jobs.sqlite").failures()
    for failure in failures:
        report.add(
            "ERROR",
            "JOB_NOT_COMPLETED",
            str(failure["stage"]),
            str(failure.get("error") or failure["status"]),
            item_id=str(failure["item_id"]),
        )

    report.coverage = {
        "videos": len(video_ids),
        "shots": len(shot_ids),
        "frames": len(frame_ids),
        "missing_frame_files": frame_path_counts["missing_file"],
        "frame_paths": {
            "total": len(frames),
            **frame_path_counts,
            "valid_ratio": frame_path_counts["valid"] / len(frames) if frames else 0.0,
        },
        "missing_video_files": missing_video_files,
        "embeddings": embedding_coverage,
        "embedding_alignment": embedding_alignment,
        "ocr_documents": sum(row.get("source") == "ocr" for row in text),
        "asr_documents": sum(row.get("source") == "asr" for row in text),
        "captioned_frames": len({str(row.get("frame_id")) for row in captions} & frame_ids),
        "job_failures": len(failures),
    }
    return report


def verify_artifact_fingerprints(payload: dict[str, object]) -> list[str]:
    """Return artifact names whose current contents differ from readiness."""
    stale: list[str] = []
    artifacts = payload.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return ["<invalid-artifact-map>"]
    for name, expected in artifacts.items():
        if not isinstance(expected, dict) or "path" not in expected:
            stale.append(str(name))
            continue
        path = Path(str(expected["path"]))
        if not path.exists() or _fingerprint(path)["sha256"] != expected.get("sha256"):
            stale.append(str(name))
    return stale


def verify_embedding_provenance(experiment: Experiment) -> list[str]:
    """Compare the current query-encoder resolution with offline metadata."""
    path = experiment.run_dir / "manifests" / "embeddings.jsonl"
    rows = JsonlManifest(path).read_all(strict=True)
    by_model = {str(row.get("model_name")): row for row in rows if row.get("model_name")}
    errors: list[str] = []
    configured = set(experiment.config.embedding_models)
    for extra in sorted(set(by_model) - configured):
        errors.append(f"unconfigured artifact model={extra}")
    for model in experiment.config.embedding_models:
        row = by_model.get(model)
        if row is None:
            errors.append(f"missing provenance model={model}")
            continue
        try:
            expected = resolve_embedding_model(model).to_dict()
        except EmbeddingError as exc:
            errors.append(str(exc))
            continue
        for provenance_field in (
            "requested_name",
            "backend",
            "resolved_model_id",
            "revision",
            "preprocessing",
        ):
            if row.get(provenance_field) != expected[provenance_field]:
                errors.append(
                    f"model={model} field={provenance_field} "
                    f"offline={row.get(provenance_field)!r} "
                    f"runtime={expected[provenance_field]!r}"
                )
    return errors


def verify_frame_files(experiment: Experiment) -> list[dict[str, str]]:
    """Recheck every frame at activation to catch post-validation deletion."""
    path = experiment.run_dir / "manifests" / "frames.jsonl"
    issues: list[dict[str, str]] = []
    for row in JsonlManifest(path).read_all(strict=True):
        frame_id = str(row.get("frame_id", ""))
        resolution = resolve_experiment_frame_path(experiment, row.get("frame_path"))
        if not resolution.valid or not resolution.canonical:
            issues.append(
                {
                    "frame_id": frame_id,
                    "reason": resolution.reason or "FRAME_PATH_NOT_CANONICAL",
                    "path": str(resolution.raw_path),
                }
            )
    return issues


def _read(
    report: ExperimentValidationReport, path: Path, stage: str, *, required: bool
) -> list[dict[str, object]]:
    if not path.exists():
        if required:
            report.add("ERROR", "MANIFEST_MISSING", stage, str(path), artifact=path)
        return []
    inspected = JsonlManifest(path).inspect()
    for corrupt in inspected.corrupt_lines:
        report.add(
            "ERROR",
            "MANIFEST_CORRUPT",
            stage,
            f"line={corrupt.line_number}: {corrupt.error}",
            artifact=path,
        )
    if required and not inspected.rows:
        report.add("ERROR", "MANIFEST_EMPTY", stage, str(path), artifact=path)
    report.artifacts[path.name] = _fingerprint(path)
    return inspected.rows


def _unique(
    report: ExperimentValidationReport,
    rows: list[dict[str, object]],
    key: str,
    stage: str,
) -> set[str]:
    values: set[str] = set()
    for row in rows:
        if key not in row:
            report.add("ERROR", "MANIFEST_FIELD_MISSING", stage, key)
            continue
        value = str(row[key])
        if value in values:
            report.add("ERROR", "DUPLICATE_ID", stage, value, item_id=value)
        values.add(value)
    return values


def _validate_embedding(
    report: ExperimentValidationReport,
    directory: Path,
    model: str,
    known_frames: set[str],
    provenance: dict[str, object] | None,
    *,
    allow_partial: bool,
) -> dict[str, object]:
    import numpy as np

    vector_file = vectors_path(directory, model)
    ids_file = frame_ids_path(directory, model)
    valid = True

    def fail(code: str, message: str) -> None:
        nonlocal valid
        valid = False
        report.add("ERROR", code, "EMBED", message, item_id=model)

    try:
        expected_spec = resolve_embedding_model(model)
    except EmbeddingError as exc:
        fail("EMBEDDING_MODEL_UNSUPPORTED", str(exc))
        expected_spec = None
    if provenance is None:
        fail("EMBEDDING_PROVENANCE_MISSING", f"model={model}")
    elif expected_spec is not None:
        expected = expected_spec.to_dict()
        for provenance_field in (
            "requested_name",
            "backend",
            "resolved_model_id",
            "revision",
            "preprocessing",
        ):
            if provenance.get(provenance_field) != expected[provenance_field]:
                fail(
                    "EMBEDDING_PROVENANCE_MISMATCH",
                    f"field={provenance_field} expected={expected[provenance_field]!r} "
                    f"actual={provenance.get(provenance_field)!r}",
                )

    if not vector_file.exists() or not ids_file.exists():
        fail("EMBEDDING_ARTIFACT_MISSING", f"model={model}")
        return {"status": "INVALID", "coverage_ratio": 0.0}
    report.artifacts[f"embedding:{model}:vectors"] = _fingerprint(vector_file)
    report.artifacts[f"embedding:{model}:ids"] = _fingerprint(ids_file)
    try:
        ids = json.loads(ids_file.read_text(encoding="utf-8"))
        vectors = np.load(vector_file)["embeddings"]
    except Exception as exc:
        fail("EMBEDDING_ARTIFACT_INVALID", str(exc))
        return {"status": "INVALID", "coverage_ratio": 0.0}
    if not isinstance(ids, list):
        fail("EMBEDDING_IDS_INVALID", "frame IDs must be a JSON list")
        ids = []
    ids = [str(value) for value in ids]
    if vectors.ndim != 2:
        fail("EMBEDDING_DIMENSION_INVALID", str(vectors.shape))
    elif provenance is not None:
        try:
            recorded_dimension = int(provenance["dimension"])
        except (KeyError, TypeError, ValueError) as exc:
            fail("EMBEDDING_PROVENANCE_INVALID", f"dimension: {exc}")
        else:
            if recorded_dimension != int(vectors.shape[1]):
                fail(
                    "EMBEDDING_PROVENANCE_DIMENSION_MISMATCH",
                    f"recorded={recorded_dimension} actual={vectors.shape[1]}",
                )
    if len(vectors) != len(ids):
        fail("EMBEDDING_ID_COUNT_MISMATCH", f"vectors={len(vectors)} ids={len(ids)}")
    if len(ids) != len(set(ids)):
        fail("EMBEDDING_DUPLICATE_FRAME_ID", model)
    if not np.isfinite(vectors).all():
        fail("EMBEDDING_NON_FINITE", model)
    unknown = set(ids) - known_frames
    missing = known_frames - set(ids)
    if unknown:
        fail("EMBEDDING_UNKNOWN_FRAME", f"count={len(unknown)}")
    if missing:
        if allow_partial:
            report.add(
                "WARNING",
                "EMBEDDING_COVERAGE_PARTIAL",
                "EMBED",
                f"model={model} missing={len(missing)}",
                item_id=model,
            )
        else:
            fail("EMBEDDING_COVERAGE_INCOMPLETE", f"missing={len(missing)}")
    covered = set(ids) & known_frames
    return {
        "status": "READY" if valid else "INVALID",
        "vector_count": len(vectors),
        "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else None,
        "coverage_ratio": len(covered) / len(known_frames) if known_frames else 0.0,
        "missing_frames": len(missing),
        "provenance": expected_spec.to_dict() if expected_spec is not None else None,
    }


def _validate_cross_model_alignment(
    report: ExperimentValidationReport,
    directory: Path,
    models: tuple[str, ...],
) -> dict[str, object]:
    ids_by_model: dict[str, list[str]] = {}
    for model in models:
        path = frame_ids_path(directory, model)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            ids_by_model[model] = [str(value) for value in payload]
    if not models or len(ids_by_model) != len(models):
        return {
            "status": "INVALID",
            "policy": "join_by_frame_id",
            "models_checked": sorted(ids_by_model),
        }

    canonical_model = models[0]
    canonical_ids = ids_by_model[canonical_model]
    canonical_set = set(canonical_ids)
    same_set = True
    same_order = True
    unique_ids = len(canonical_ids) == len(canonical_set)
    common_ids = canonical_set.copy()
    for model in models[1:]:
        model_ids = ids_by_model[model]
        model_set = set(model_ids)
        unique_ids = unique_ids and len(model_ids) == len(model_set)
        common_ids &= model_set
        if model_set != canonical_set:
            same_set = False
            report.add(
                "ERROR",
                "EMBEDDING_CROSS_MODEL_SET_MISMATCH",
                "EMBED",
                f"canonical={canonical_model} model={model} "
                f"missing={len(canonical_set - model_set)} extra={len(model_set - canonical_set)}",
                item_id=model,
            )
        if model_ids != canonical_ids:
            same_order = False
    return {
        "status": "READY" if same_set and unique_ids and canonical_ids else "INVALID",
        "policy": "join_by_frame_id",
        "canonical_model": canonical_model,
        "canonical_frame_count": len(canonical_ids),
        "common_frame_count": len(common_ids),
        "same_frame_id_set": same_set,
        "unique_frame_ids": unique_ids,
        "same_row_order": same_order,
        "models_checked": list(models),
    }


def _fingerprint(path: Path) -> dict[str, object]:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }
