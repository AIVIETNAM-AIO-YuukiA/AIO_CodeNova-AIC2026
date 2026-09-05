import json

import pytest

from cli.main import artifact_overrides_from_args, build_parser, config_from_args
from config.settings import Experiment, PipelineConfig
from core.errors import ExperimentConfigError


def test_open_restores_artifact_config_and_keeps_runtime_controls(tmp_path):
    offline_config = PipelineConfig(
        runs_dir=tmp_path,
        embedding_models=(
            "jina-clip-v2",
            "siglip2-so400m",
        ),
    )

    created = Experiment.create(
        config=offline_config,
        name="offline-online-mismatch",
    )

    runtime_config = PipelineConfig(
        runs_dir=tmp_path,
        embedding_models=("beit3",),
        top_k=99,
        device="cpu",
    )

    reopened = Experiment.open(
        config=runtime_config,
        name=created.name,
    )

    persisted = json.loads((created.run_dir / "config.json").read_text(encoding="utf-8"))
    persisted_models = tuple(persisted["config"]["embedding_models"])

    assert persisted_models == (
        "jina-clip-v2",
        "siglip2-so400m",
    )

    assert reopened.config.embedding_models == persisted_models
    assert reopened.config.top_k == 99
    assert reopened.config.device == "cpu"
    assert reopened.config.config_hash() == offline_config.config_hash()


def test_open_rejects_missing_experiment_metadata(tmp_path):
    run_dir = tmp_path / "missing-metadata"
    run_dir.mkdir()

    with pytest.raises(ExperimentConfigError, match="metadata is missing"):
        Experiment.open(PipelineConfig(runs_dir=tmp_path), "missing-metadata")


def test_create_resume_does_not_overwrite_persisted_artifact_config(tmp_path):
    original = PipelineConfig(runs_dir=tmp_path, embedding_models=("jina-clip-v2",))
    Experiment.create(original, name="resume-config")
    metadata_path = tmp_path / "resume-config" / "config.json"
    before = metadata_path.read_text(encoding="utf-8")

    resumed = Experiment.create(
        PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3",), device="cpu"),
        name="resume-config",
        resume=True,
    )

    assert resumed.config.embedding_models == ("jina-clip-v2",)
    assert resumed.config.device == "cpu"
    assert metadata_path.read_text(encoding="utf-8") == before


def test_open_rejects_explicit_artifact_override_mismatch(tmp_path):
    Experiment.create(
        PipelineConfig(runs_dir=tmp_path, embedding_models=("jina-clip-v2",)),
        name="strict-config",
    )

    with pytest.raises(ExperimentConfigError, match="embedding_models") as captured:
        Experiment.open(
            PipelineConfig(runs_dir=tmp_path),
            "strict-config",
            artifact_overrides={"embedding_models": ("beit3",)},
        )

    assert "persisted=('jina-clip-v2',)" in str(captured.value)
    assert "requested=('beit3',)" in str(captured.value)


def test_open_rejects_corrupt_metadata_and_config_hash(tmp_path):
    experiment = Experiment.create(PipelineConfig(runs_dir=tmp_path), name="corrupt-config")
    metadata_path = experiment.run_dir / "config.json"
    metadata_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="metadata is invalid"):
        Experiment.open(PipelineConfig(runs_dir=tmp_path), experiment.name)

    experiment = Experiment.create(PipelineConfig(runs_dir=tmp_path), name="bad-hash")
    metadata_path = experiment.run_dir / "config.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["config_hash"] = "00000000"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="config_hash mismatch"):
        Experiment.open(PipelineConfig(runs_dir=tmp_path), experiment.name)


def test_open_reads_legacy_unversioned_metadata_without_rewriting_it(tmp_path):
    experiment = Experiment.create(PipelineConfig(runs_dir=tmp_path), name="legacy-config")
    metadata_path = experiment.run_dir / "config.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    payload.pop("artifact_config")
    payload.pop("runtime_defaults")
    metadata_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    before = metadata_path.read_text(encoding="utf-8")

    reopened = Experiment.open(PipelineConfig(runs_dir=tmp_path, device="cpu"), experiment.name)

    assert reopened.config.embedding_models == ("jina-clip-v2",)
    assert reopened.config.device == "cpu"
    assert metadata_path.read_text(encoding="utf-8") == before


def test_existing_run_ignores_environment_artifact_defaults(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODELS", "beit3")
    args = build_parser().parse_args(
        ["validate-index", "--experiment-name", "existing", "--device", "cpu"]
    )

    assert config_from_args(args).embedding_models == ("beit3",)
    assert artifact_overrides_from_args(args) == {}


def test_ui_displays_active_persisted_experiment_and_models(tmp_path):
    from ui.server import _render_index_html

    experiment = Experiment(
        name="competition-final",
        run_dir=tmp_path,
        config=PipelineConfig(
            runs_dir=tmp_path,
            embedding_models=("jina-clip-v2", "siglip2-so400m"),
        ),
    )

    rendered = _render_index_html(experiment, 37)

    assert "Experiment: competition-final" in rendered
    assert "Models: jina-clip-v2, siglip2-so400m" in rendered
    assert 'value="37"' in rendered


def test_config_hash_excludes_runtime_controls_but_includes_artifact_fields(tmp_path):
    base = PipelineConfig(runs_dir=tmp_path, top_k=20, device="auto")
    runtime_changed = PipelineConfig(runs_dir=tmp_path / "other", top_k=99, device="cpu")
    artifact_changed = PipelineConfig(
        runs_dir=tmp_path, embedding_models=("jina-clip-v2", "siglip2-so400m")
    )

    assert base.config_hash() == runtime_changed.config_hash()
    assert base.config_hash() != artifact_changed.config_hash()


def test_open_rejects_future_metadata_schema(tmp_path):
    experiment = Experiment.create(PipelineConfig(runs_dir=tmp_path), name="future-schema")
    metadata_path = experiment.run_dir / "config.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="unsupported schema_version=999"):
        Experiment.open(PipelineConfig(runs_dir=tmp_path), experiment.name)
