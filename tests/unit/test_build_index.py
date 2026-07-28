"""Tests for the build-index stage (per-model embedding file loading + merge)."""

import json

import numpy as np

from config.settings import Experiment, PipelineConfig
from indexing.build_index import build_index


class FakeVectorIndex:
    def __init__(self, *args, **kwargs) -> None:
        self.built: tuple[dict[str, list[list[float]]], list[str]] | None = None

    def build(self, embeddings_by_model, frame_ids) -> None:
        self.built = (embeddings_by_model, frame_ids)


def _write_embeddings(embeddings_dir, model_name: str, ids: list[str], vectors) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_dir / f"frames__{model_name}.npz", embeddings=np.array(vectors))
    (embeddings_dir / f"frame_ids__{model_name}.json").write_text(json.dumps(ids))


def test_build_index_single_model_reads_named_file(tmp_path, monkeypatch) -> None:
    config = PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3",))
    experiment = Experiment(name="exp", run_dir=tmp_path, config=config)
    _write_embeddings(tmp_path / "embeddings", "beit3", ["f1", "f2"], [[1.0, 0.0], [0.0, 1.0]])

    fake_index = FakeVectorIndex()
    monkeypatch.setattr("indexing.build_index.build_vector_index", lambda exp: fake_index)

    count = build_index(experiment)

    assert count == 2
    embeddings_by_model, frame_ids = fake_index.built
    assert frame_ids == ["f1", "f2"]
    assert embeddings_by_model["beit3"] == [[1.0, 0.0], [0.0, 1.0]]


def test_build_index_multi_model_intersects_common_frames(tmp_path, monkeypatch) -> None:
    config = PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3", "siglip2"))
    experiment = Experiment(name="exp", run_dir=tmp_path, config=config)
    embeddings_dir = tmp_path / "embeddings"
    # siglip2 is missing "f3" (e.g. its embed-frames run is behind)
    _write_embeddings(embeddings_dir, "beit3", ["f1", "f2", "f3"], [[1, 0], [0, 1], [1, 1]])
    _write_embeddings(embeddings_dir, "siglip2", ["f1", "f2"], [[2, 0], [0, 2]])

    fake_index = FakeVectorIndex()
    monkeypatch.setattr("indexing.build_index.build_vector_index", lambda exp: fake_index)

    count = build_index(experiment)

    assert count == 2  # only f1, f2 are embedded by both models
    embeddings_by_model, frame_ids = fake_index.built
    assert frame_ids == ["f1", "f2"]
    assert embeddings_by_model["beit3"] == [[1, 0], [0, 1]]
    assert embeddings_by_model["siglip2"] == [[2, 0], [0, 2]]


def test_build_index_falls_back_to_in_progress_checkpoint(tmp_path, monkeypatch) -> None:
    """A model with no finished frames__<model>.npz yet (embed-frames still
    running) should still be indexable from its *.checkpoint.npz/.json."""
    config = PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3",))
    experiment = Experiment(name="exp", run_dir=tmp_path, config=config)
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir(parents=True)
    np.savez_compressed(
        embeddings_dir / "frames__beit3.checkpoint.npz", embeddings=np.array([[1.0, 0.0]])
    )
    (embeddings_dir / "frame_ids__beit3.checkpoint.json").write_text(json.dumps(["f1"]))

    fake_index = FakeVectorIndex()
    monkeypatch.setattr("indexing.build_index.build_vector_index", lambda exp: fake_index)

    count = build_index(experiment)

    assert count == 1
    embeddings_by_model, frame_ids = fake_index.built
    assert frame_ids == ["f1"]
    assert embeddings_by_model["beit3"] == [[1.0, 0.0]]


def test_build_index_prefers_finished_file_over_checkpoint(tmp_path, monkeypatch) -> None:
    config = PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3",))
    experiment = Experiment(name="exp", run_dir=tmp_path, config=config)
    embeddings_dir = tmp_path / "embeddings"
    _write_embeddings(embeddings_dir, "beit3", ["f1", "f2"], [[1.0, 0.0], [0.0, 1.0]])
    # Stale leftover checkpoint from an earlier interrupted run; must be ignored.
    np.savez_compressed(
        embeddings_dir / "frames__beit3.checkpoint.npz", embeddings=np.array([[9.0, 9.0]])
    )
    (embeddings_dir / "frame_ids__beit3.checkpoint.json").write_text(json.dumps(["stale"]))

    fake_index = FakeVectorIndex()
    monkeypatch.setattr("indexing.build_index.build_vector_index", lambda exp: fake_index)

    build_index(experiment)

    embeddings_by_model, frame_ids = fake_index.built
    assert frame_ids == ["f1", "f2"]
