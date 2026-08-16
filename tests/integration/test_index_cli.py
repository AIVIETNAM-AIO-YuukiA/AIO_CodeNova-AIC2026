import json
from types import SimpleNamespace

import cli.main as cli
from cli.main import main
from config.settings import Experiment, PipelineConfig
from indexing.manifest import JsonlManifest


def test_preflight_cli_creates_approved_plan_and_execution_log(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.mp4").write_bytes(b"video")
    runs_dir = tmp_path / "runs"

    result = main(
        [
            "preflight-index",
            "--experiment-name",
            "test-exp",
            "--input",
            str(input_dir),
            "--runs-dir",
            str(runs_dir),
            "--device",
            "cpu",
            "--approve",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "APPROVED"
    assert payload["dataset"]["video_count"] == 1
    assert list((runs_dir / "test-exp" / "logs" / "executions").glob("*.log"))


def test_validate_index_cli_writes_invalid_readiness_for_missing_artifacts(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.mp4").write_bytes(b"video")
    runs_dir = tmp_path / "runs"
    assert (
        main(
            [
                "preflight-index",
                "--experiment-name",
                "test-exp",
                "--input",
                str(input_dir),
                "--runs-dir",
                str(runs_dir),
                "--device",
                "cpu",
                "--approve",
            ]
        )
        == 0
    )
    capsys.readouterr()

    result = main(
        [
            "validate-index",
            "--experiment-name",
            "test-exp",
            "--runs-dir",
            str(runs_dir),
            "--device",
            "cpu",
        ]
    )

    assert result == 1
    readiness = json.loads((runs_dir / "test-exp" / "readiness.json").read_text(encoding="utf-8"))
    assert readiness["status"] == "INVALID"
    assert "MANIFEST_MISSING" in {issue["code"] for issue in readiness["issues"]}


def test_offline_index_runs_all_vector_stages_then_quality_gate(tmp_path, monkeypatch, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.mp4").write_bytes(b"video")
    calls = []

    monkeypatch.setattr(cli, "ingest_videos", lambda *a, **k: calls.append("ingest"))
    monkeypatch.setattr(cli, "detect_shots", lambda *a, **k: calls.append("shots"))
    monkeypatch.setattr(cli, "extract_frames", lambda *a, **k: calls.append("frames"))
    monkeypatch.setattr(cli, "embed_frames", lambda *a, **k: calls.append("embed"))
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: calls.append("index"))
    report = SimpleNamespace(status="READY", to_dict=lambda: {"status": "READY"})
    monkeypatch.setattr(
        cli,
        "validate_experiment_artifacts",
        lambda experiment: calls.append("validate") or report,
    )
    monkeypatch.setattr(
        cli,
        "write_readiness",
        lambda experiment, value: experiment.run_dir / "readiness.json",
    )

    result = main(
        [
            "offline-index",
            "--experiment-name",
            "competition-run",
            "--input",
            str(input_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
            "--approve",
        ]
    )

    assert result == 0
    assert calls == ["ingest", "shots", "frames", "embed", "index", "validate"]
    assert '"status": "READY"' in capsys.readouterr().out


def test_offline_index_without_approval_only_writes_pending_plan(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        cli,
        "ingest_videos",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("indexing started")),
    )

    result = main(
        [
            "offline-index",
            "--experiment-name",
            "pending-run",
            "--input",
            str(input_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
        ]
    )

    assert result == 1


def test_repair_manifest_is_dry_run_by_default_and_audited(tmp_path, capsys):
    experiment = Experiment.create(
        PipelineConfig(runs_dir=tmp_path, device="cpu"), name="repair-run"
    )
    manifest = experiment.run_dir / "manifests" / "videos.jsonl"
    manifest.parent.mkdir(parents=True)
    original = '{"video_id": "ok"}\n{broken}\n'
    manifest.write_text(original, encoding="utf-8")

    assert (
        main(
            [
                "repair-manifest",
                "videos",
                "--experiment-name",
                experiment.name,
                "--runs-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["mode"] == "DRY_RUN"
    assert len(dry_run["corrupt_lines"]) == 1
    assert manifest.read_text(encoding="utf-8") == original

    assert (
        main(
            [
                "repair-manifest",
                "videos",
                "--experiment-name",
                experiment.name,
                "--runs-dir",
                str(tmp_path),
                "--device",
                "cpu",
                "--apply",
            ]
        )
        == 0
    )
    applied = json.loads(capsys.readouterr().out)
    assert applied["changed"] is True
    assert manifest.read_text(encoding="utf-8") == '{"video_id": "ok"}\n'
    assert list(manifest.parent.glob("videos.jsonl.*.bak"))
    assert list((experiment.run_dir / "logs" / "repairs").glob("*.json"))


def test_migrate_frame_paths_cli_is_dry_run_then_apply(tmp_path, capsys):
    experiment = Experiment.create(PipelineConfig(runs_dir=tmp_path, device="cpu"), name="path-run")
    frame = experiment.run_dir / "frames" / "v1" / "f1.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    manifest = experiment.run_dir / "manifests" / "frames.jsonl"
    JsonlManifest(manifest).replace_all(
        [{"frame_id": "f1", "frame_path": str(frame)}], unique_key="frame_id"
    )
    common = [
        "migrate-frame-paths",
        "--experiment-name",
        experiment.name,
        "--runs-dir",
        str(tmp_path),
        "--legacy-root",
        str(tmp_path),
        "--device",
        "cpu",
    ]

    assert main(common) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["mode"] == "DRY_RUN"
    assert dry_run["changed_records"] == 1
    assert JsonlManifest(manifest).read_all(strict=True)[0]["frame_path"] == str(frame)

    assert main([*common, "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["mode"] == "APPLY"
    assert applied["changed"] is True
    assert JsonlManifest(manifest).read_all(strict=True)[0]["frame_path"] == "frames/v1/f1.jpg"


def test_offline_index_rejects_unknown_model_before_writing_artifacts(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.mp4").write_bytes(b"video")
    runs_dir = tmp_path / "runs"

    result = main(
        [
            "offline-index",
            "--experiment-name",
            "unknown-model-run",
            "--input",
            str(input_dir),
            "--runs-dir",
            str(runs_dir),
            "--embedding-models",
            "unknown-model",
            "--device",
            "cpu",
            "--approve",
        ]
    )

    assert result == 2
    assert "Unsupported embedding model" in capsys.readouterr().err
    run_dir = runs_dir / "unknown-model-run"
    assert not (run_dir / "embeddings").exists()
    assert not (run_dir / "manifests" / "embeddings.jsonl").exists()


def test_existing_run_cli_rejects_explicit_artifact_config_mismatch(tmp_path, capsys):
    Experiment.create(
        PipelineConfig(runs_dir=tmp_path, embedding_models=("jina-clip-v2",)),
        name="persisted-cli-config",
    )

    result = main(
        [
            "validate-index",
            "--experiment-name",
            "persisted-cli-config",
            "--runs-dir",
            str(tmp_path),
            "--embedding-models",
            "beit3",
            "--device",
            "cpu",
        ]
    )

    assert result == 2
    error = capsys.readouterr().err
    assert "Artifact config mismatch for embedding_models" in error
    assert "persisted=('jina-clip-v2',)" in error
    assert "requested=('beit3',)" in error
