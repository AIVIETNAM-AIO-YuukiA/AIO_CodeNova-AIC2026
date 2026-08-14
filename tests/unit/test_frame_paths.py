from pathlib import Path

from config.settings import Experiment, PipelineConfig
from core.paths import canonical_frame_path, resolve_experiment_frame_path
from indexing.frame_paths import apply_frame_path_migration, plan_frame_path_migration
from indexing.manifest import JsonlManifest
from indexing.validation import verify_frame_files


def _experiment(tmp_path: Path) -> Experiment:
    return Experiment("exp", tmp_path / "runs" / "exp", PipelineConfig(runs_dir=tmp_path / "runs"))


def test_frame_path_is_resolved_against_experiment_not_cwd(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    frame = experiment.run_dir / "frames" / "v1" / "f1.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolution = resolve_experiment_frame_path(experiment, "frames/v1/f1.jpg")

    assert resolution.valid
    assert resolution.canonical
    assert resolution.resolved_path == frame.resolve()
    assert canonical_frame_path(experiment, frame) == "frames/v1/f1.jpg"


def test_frame_path_rejects_missing_and_outside_files(tmp_path):
    experiment = _experiment(tmp_path)
    experiment.run_dir.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")

    assert (
        resolve_experiment_frame_path(experiment, "frames/missing.jpg").reason
        == "FRAME_FILE_MISSING"
    )
    assert (
        resolve_experiment_frame_path(experiment, outside).reason == "FRAME_PATH_OUTSIDE_EXPERIMENT"
    )


def test_migration_updates_aggregate_and_partitions_and_invalidates_readiness(tmp_path):
    experiment = _experiment(tmp_path)
    frame = experiment.run_dir / "frames" / "v1" / "f1.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    legacy = str(frame)
    aggregate = experiment.run_dir / "manifests" / "frames.jsonl"
    partition = experiment.run_dir / "manifests" / "partitions" / "frames" / "v1.jsonl"
    row = {"frame_id": "f1", "video_id": "v1", "shot_id": "s1", "frame_path": legacy}
    JsonlManifest(aggregate).replace_all([row], unique_key="frame_id")
    JsonlManifest(partition).replace_all([row], unique_key="frame_id")
    readiness = experiment.run_dir / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")

    plan = plan_frame_path_migration(experiment, tmp_path)

    assert plan.issues == []
    assert plan.changed_records == 2
    assert JsonlManifest(aggregate).read_all(strict=True)[0]["frame_path"] == legacy
    audit = apply_frame_path_migration(experiment, plan)
    assert JsonlManifest(aggregate).read_all(strict=True)[0]["frame_path"] == "frames/v1/f1.jpg"
    assert JsonlManifest(partition).read_all(strict=True)[0]["frame_path"] == "frames/v1/f1.jpg"
    assert not readiness.exists()
    assert audit.is_file()
    assert list(aggregate.parent.glob("frames.jsonl.*.bak"))
    assert list(partition.parent.glob("v1.jsonl.*.bak"))


def test_activation_recheck_detects_frame_deleted_after_validation(tmp_path):
    experiment = _experiment(tmp_path)
    frame = experiment.run_dir / "frames" / "f1.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl").replace_all(
        [{"frame_id": "f1", "frame_path": "frames/f1.jpg"}], unique_key="frame_id"
    )
    assert verify_frame_files(experiment) == []

    frame.unlink()

    assert verify_frame_files(experiment) == [
        {"frame_id": "f1", "reason": "FRAME_FILE_MISSING", "path": "frames/f1.jpg"}
    ]
