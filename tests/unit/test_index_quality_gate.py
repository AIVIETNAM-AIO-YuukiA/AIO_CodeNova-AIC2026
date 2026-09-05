import json

import numpy as np
import pytest
import logging
from hashlib import sha256

from config.settings import Experiment, PipelineConfig
from core.logging import configure_logging
from indexing.embedding_paths import frame_ids_path, vectors_path
from indexing.manifest import JsonlManifest, ManifestError
from indexing.partitions import PartitionStore
from indexing.preflight import PreflightError, build_preflight_plan
from indexing.readiness import write_readiness
from indexing.validation import validate_experiment_artifacts, verify_artifact_fingerprints
from indexing.validation import ExperimentValidationReport, _validate_cross_model_alignment


def _experiment(tmp_path):
    return Experiment(
        name="exp",
        run_dir=tmp_path,
        config=PipelineConfig(runs_dir=tmp_path),
    )


def test_strict_manifest_rejects_corrupt_line(tmp_path):
    path = tmp_path / "frames.jsonl"
    path.write_text('{"frame_id":"f1"}\n{"frame_id":\n', encoding="utf-8")

    with pytest.raises(ManifestError):
        JsonlManifest(path).read_all(strict=True)

    result = JsonlManifest(path).inspect()
    assert [row["frame_id"] for row in result.rows] == ["f1"]
    assert result.corrupt_lines[0].line_number == 2


def test_invalid_partition_does_not_replace_previous_data(tmp_path):
    store = PartitionStore(tmp_path / "partitions")
    store.write(
        "v1",
        [{"video_id": "v1", "shot_id": "v1_s1"}],
        partition_key="video_id",
        unique_key="shot_id",
    )
    previous = store.path_for("v1").read_text(encoding="utf-8")

    with pytest.raises(ManifestError):
        store.write(
            "v1",
            [
                {"video_id": "v1", "shot_id": "dup"},
                {"video_id": "v1", "shot_id": "dup"},
            ],
            partition_key="video_id",
            unique_key="shot_id",
        )

    assert store.path_for("v1").read_text(encoding="utf-8") == previous


def test_quality_gate_detects_stale_readiness_and_embedding_mismatch(tmp_path):
    experiment = _experiment(tmp_path)
    manifests = tmp_path / "manifests"
    frame_file = tmp_path / "frames" / "v1" / "v1_s1_f1.jpg"
    frame_file.parent.mkdir(parents=True)
    frame_file.write_bytes(b"frame")
    video_file = tmp_path / "v1.mp4"
    video_file.write_bytes(b"v")

    JsonlManifest(manifests / "videos.jsonl").replace_all(
        [
            {
                "video_id": "v1",
                "path": str(video_file),
                "checksum": sha256(b"v").hexdigest(),
                "size_bytes": 1,
            }
        ],
        unique_key="video_id",
    )
    JsonlManifest(manifests / "shots.jsonl").replace_all(
        [{"video_id": "v1", "shot_id": "v1_s1", "start_frame": 0, "end_frame": 1}],
        unique_key="shot_id",
    )
    JsonlManifest(manifests / "frames.jsonl").replace_all(
        [
            {
                "video_id": "v1",
                "shot_id": "v1_s1",
                "frame_id": "v1_s1_f1",
                "frame_path": "frames/v1/v1_s1_f1.jpg",
            }
        ],
        unique_key="frame_id",
    )
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    np.savez_compressed(vectors_path(embedding_dir, "jina-clip-v2"), embeddings=np.ones((1, 2)))
    frame_ids_path(embedding_dir, "jina-clip-v2").write_text(
        json.dumps(["v1_s1_f1"]), encoding="utf-8"
    )
    JsonlManifest(manifests / "embeddings.jsonl").replace_all(
        [
            {
                "model_name": "jina-clip-v2",
                "requested_name": "jina-clip-v2",
                "backend": "JinaClipEmbedder",
                "resolved_model_id": "jinaai/jina-clip-v2",
                "revision": None,
                "preprocessing": "jina-clip-v2:image-512:l2-normalized",
                "dimension": 2,
            }
        ],
        unique_key="model_name",
    )

    report = validate_experiment_artifacts(experiment)
    assert report.status == "READY"
    provenance_path = manifests / "embeddings.jsonl"
    provenance_text = provenance_path.read_text(encoding="utf-8")
    provenance = json.loads(provenance_text)
    provenance["backend"] = "SiglipEmbedder"
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")
    mismatch = validate_experiment_artifacts(experiment)
    assert "EMBEDDING_PROVENANCE_MISMATCH" in {issue.code for issue in mismatch.issues}
    provenance_path.write_text(provenance_text, encoding="utf-8")
    readiness_path = write_readiness(experiment, report)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert verify_artifact_fingerprints(readiness) == []

    video_file.write_bytes(b"changed")
    assert verify_artifact_fingerprints(readiness) == ["video-source:v1"]
    video_file.write_bytes(b"v")
    (manifests / "frames.jsonl").write_text("", encoding="utf-8")
    assert verify_artifact_fingerprints(readiness) == ["frames.jsonl"]


def test_quality_gate_detects_changed_source_video(tmp_path):
    experiment = _experiment(tmp_path)
    manifests = tmp_path / "manifests"
    video_file = tmp_path / "v1.mp4"
    video_file.write_bytes(b"original")
    JsonlManifest(manifests / "videos.jsonl").replace_all(
        [
            {
                "video_id": "v1",
                "path": str(video_file),
                "checksum": sha256(b"original").hexdigest(),
                "size_bytes": len(b"original"),
            }
        ]
    )
    JsonlManifest(manifests / "shots.jsonl").replace_all([])
    JsonlManifest(manifests / "frames.jsonl").replace_all([])
    video_file.write_bytes(b"modified")

    report = validate_experiment_artifacts(experiment)

    assert "VIDEO_CHECKSUM_MISMATCH" in {issue.code for issue in report.issues}


def test_quality_gate_marks_vector_id_count_mismatch_invalid(tmp_path):
    experiment = _experiment(tmp_path)
    manifests = tmp_path / "manifests"
    frame_file = tmp_path / "f.jpg"
    frame_file.write_bytes(b"frame")
    JsonlManifest(manifests / "videos.jsonl").replace_all(
        [{"video_id": "v1", "path": "v1.mp4", "checksum": "x", "size_bytes": 1}]
    )
    JsonlManifest(manifests / "shots.jsonl").replace_all(
        [{"video_id": "v1", "shot_id": "s1", "start_frame": 0, "end_frame": 1}]
    )
    JsonlManifest(manifests / "frames.jsonl").replace_all(
        [{"video_id": "v1", "shot_id": "s1", "frame_id": "f1", "frame_path": str(frame_file)}]
    )
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    np.savez_compressed(vectors_path(embedding_dir, "jina-clip-v2"), embeddings=np.ones((2, 2)))
    frame_ids_path(embedding_dir, "jina-clip-v2").write_text(json.dumps(["f1"]), encoding="utf-8")

    report = validate_experiment_artifacts(experiment)

    assert report.status == "INVALID"
    assert "EMBEDDING_ID_COUNT_MISMATCH" in {issue.code for issue in report.issues}
    assert report.coverage["embeddings"]["jina-clip-v2"]["status"] == "INVALID"


def test_preflight_reports_dataset_config_and_resolved_device(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.mp4").write_bytes(b"video")
    experiment = _experiment(tmp_path / "run")

    monkeypatch.setattr(
        "indexing.preflight._device_info",
        lambda requested: {
            "requested": requested,
            "resolved": "cuda:0",
            "cuda_available": True,
            "gpu_name": "Fake GPU",
        },
    )

    plan = build_preflight_plan(experiment, input_dir, approved=False)

    assert plan["status"] == "PENDING_APPROVAL"
    assert plan["dataset"]["video_count"] == 1
    assert plan["dataset"]["total_size_bytes"] == 5
    assert plan["device"]["requested"] == "auto"
    assert plan["device"]["resolved"] == "cuda:0"
    assert plan["config"]["embedding_models"] == ("jina-clip-v2",)


def test_logging_mirrors_execution_to_aggregate_and_per_run_file(tmp_path):
    execution_id = configure_logging(tmp_path / "logs", command="embed-frames", experiment="exp")
    logging.getLogger("test").info("path=dataset/video.mp4 [model 1/2] [batch 3/4]")
    for handler in logging.getLogger().handlers:
        handler.flush()

    aggregate = (tmp_path / "logs" / "pipeline.log").read_text(encoding="utf-8")
    execution = (tmp_path / "logs" / "executions" / f"{execution_id}.log").read_text(
        encoding="utf-8"
    )
    assert execution_id in aggregate
    assert "path=dataset/video.mp4" in aggregate
    assert "[model 1/2] [batch 3/4]" in execution


def test_preflight_rejects_unknown_embedding_backend(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.mp4").write_bytes(b"video")
    experiment = Experiment(
        name="exp",
        run_dir=tmp_path / "run",
        config=PipelineConfig(embedding_models=("unknown-model",)),
    )
    monkeypatch.setattr(
        "indexing.preflight._device_info",
        lambda requested: {"requested": requested, "resolved": "cpu"},
    )

    with pytest.raises(PreflightError):
        build_preflight_plan(experiment, input_dir, approved=False)


def test_quality_gate_reports_order_difference_as_safe_frame_id_join(tmp_path):
    directory = tmp_path / "embeddings"
    directory.mkdir()
    frame_ids_path(directory, "m1").write_text(json.dumps(["f1", "f2"]), encoding="utf-8")
    frame_ids_path(directory, "m2").write_text(json.dumps(["f2", "f1"]), encoding="utf-8")
    report = ExperimentValidationReport("exp")

    alignment = _validate_cross_model_alignment(report, directory, ("m1", "m2"))

    assert alignment["status"] == "READY"
    assert alignment["same_frame_id_set"] is True
    assert alignment["same_row_order"] is False
    assert alignment["policy"] == "join_by_frame_id"
    assert report.status == "READY"


def test_quality_gate_rejects_cross_model_frame_id_set_mismatch(tmp_path):
    directory = tmp_path / "embeddings"
    directory.mkdir()
    frame_ids_path(directory, "m1").write_text(json.dumps(["f1", "f2"]), encoding="utf-8")
    frame_ids_path(directory, "m2").write_text(json.dumps(["f1"]), encoding="utf-8")
    report = ExperimentValidationReport("exp")

    alignment = _validate_cross_model_alignment(report, directory, ("m1", "m2"))

    assert alignment["status"] == "INVALID"
    assert alignment["common_frame_count"] == 1
    assert "EMBEDDING_CROSS_MODEL_SET_MISMATCH" in {issue.code for issue in report.issues}
