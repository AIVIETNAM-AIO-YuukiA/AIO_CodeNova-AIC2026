"""Tests for temporal search."""

import json

import numpy as np
import pytest

from config.settings import Experiment, PipelineConfig
from core.errors import RetrievalError
from retrieval.temporal_search import (
    temporal_search_forward,
    temporal_search_backward,
    find_segments,
    gather_frame_s,
    load_temporal_data,
)


def test_temporal_search_forward_stops_at_shot_boundary() -> None:
    """3 frame đầu similarity cao với start, 3 frame sau thấp."""
    rng = np.random.RandomState(42)
    base = rng.randn(64).astype("float32")
    base = base / np.linalg.norm(base)

    same = np.array([base + rng.randn(64).astype("float32") * 0.01 for _ in range(3)])
    same = same / np.linalg.norm(same, axis=1, keepdims=True)
    different = rng.randn(3, 64).astype("float32")
    different = different / np.linalg.norm(different, axis=1, keepdims=True)

    emb = np.concatenate([same, different], axis=0)

    end = temporal_search_forward(0, emb, tolerance_threshold=2)
    assert end == 2


def test_temporal_search_backward_stops_at_boundary() -> None:
    """Start ở vị trí 5, 3 frame trước similarity cao."""
    rng = np.random.RandomState(42)
    base = rng.randn(64).astype("float32")
    base = base / np.linalg.norm(base)

    same = np.array([base + rng.randn(64).astype("float32") * 0.01 for _ in range(3)])
    same = same / np.linalg.norm(same, axis=1, keepdims=True)
    different = rng.randn(3, 64).astype("float32")
    different = different / np.linalg.norm(different, axis=1, keepdims=True)

    emb = np.concatenate([different, same], axis=0)

    start = temporal_search_backward(5, emb, tolerance_threshold=2)
    assert start == 3


def test_find_segments_merges_overlapping() -> None:
    """2 start indices gần nhau → merged thành 1 segment."""
    rng = np.random.RandomState(42)
    base = rng.randn(64).astype("float32")
    base = base / np.linalg.norm(base)

    n = 10
    emb = np.tile(base, (n, 1))
    emb += rng.randn(n, 64).astype("float32") * 0.01
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    segments = find_segments([2, 5], emb, tolerance_threshold=2, min_gap=2)
    assert len(segments) == 1


def test_find_segments_separate() -> None:
    """2 start indices xa nhau → 2 segment riêng."""
    rng = np.random.RandomState(42)
    base1 = rng.randn(64).astype("float32")
    base2 = rng.randn(64).astype("float32")
    base1 = base1 / np.linalg.norm(base1)
    base2 = base2 / np.linalg.norm(base2)

    emb1 = np.tile(base1, (3, 1))
    filler = rng.randn(3, 64).astype("float32")
    filler = filler / np.linalg.norm(filler, axis=1, keepdims=True)
    emb2 = np.tile(base2, (3, 1))

    emb = np.concatenate([emb1, filler, emb2], axis=0)

    segments = find_segments([1, 7], emb, tolerance_threshold=1, min_gap=3)
    assert len(segments) >= 2


def test_find_segments_does_not_merge_different_shots() -> None:
    embeddings = np.ones((6, 8), dtype="float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    records = [
        {
            "frame_id": f"f{index}",
            "video_id": "v1",
            "shot_id": "s1" if index < 3 else "s2",
        }
        for index in range(6)
    ]

    segments = find_segments(
        [2, 3],
        embeddings,
        frame_records=records,
        tolerance_threshold=2,
        min_gap=2,
    )

    assert len(segments) == 2
    assert (segments[0]["start_pos"], segments[0]["end_pos"]) == (0, 2)
    assert (segments[1]["start_pos"], segments[1]["end_pos"]) == (3, 5)


def test_gather_frame_s_returns_shot_input() -> None:
    records = [
        {
            "frame_id": f"v1/f{i:04d}",
            "video_id": "v1",
            "frame_index": i,
            "timestamp_sec": float(i * 2),
            "frame_path": f"frames/v1/f{i:04d}.jpg",
        }
        for i in range(5)
    ]
    segment = {"start_pos": 1, "end_pos": 3, "length": 3, "center_idx": 2}

    shot = gather_frame_s(segment, records)
    assert shot is not None
    assert shot.frame_count == 3
    assert shot.video_id == "v1"
    assert len(shot.frame_paths) == 3


def test_gather_frame_s_none_on_empty() -> None:
    assert gather_frame_s(None, []) is None


def test_temporal_search_stops_at_different_video() -> None:
    rng = np.random.RandomState(42)
    emb = rng.randn(5, 64).astype("float32")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    records = [
        {"video_id": "v1"},
        {"video_id": "v1"},
        {"video_id": "v2"},
        {"video_id": "v2"},
        {"video_id": "v2"},
    ]

    # Forward search starting at 0 should stop at 1
    end = temporal_search_forward(0, emb, frame_records=records, tolerance_threshold=2)
    assert end == 1

    # Backward search starting at 4 should stop at 2
    start = temporal_search_backward(4, emb, frame_records=records, tolerance_threshold=2)
    assert start == 2


def _write_embeddings(embeddings_dir, model_name: str, ids: list[str], vectors) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_dir / f"frames__{model_name}.npz", embeddings=vectors)
    (embeddings_dir / f"frame_ids__{model_name}.json").write_text(json.dumps(ids))


def test_load_temporal_data_uses_first_configured_model(tmp_path) -> None:
    config = PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3", "siglip2"))
    experiment = Experiment(name="exp", run_dir=tmp_path, config=config)
    embeddings_dir = tmp_path / "embeddings"
    _write_embeddings(embeddings_dir, "beit3", ["f1", "f2"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    _write_embeddings(embeddings_dir, "siglip2", ["f1", "f2"], np.array([[9.0, 9.0], [9.0, 9.0]]))

    vectors, records = load_temporal_data(experiment)

    assert vectors.shape == (2, 2)
    assert vectors[0].tolist() == [1.0, 0.0]  # loaded beit3's file, not siglip2's
    assert [r["frame_id"] for r in records] == ["f1", "f2"]


def test_load_temporal_data_missing_file_raises(tmp_path) -> None:
    config = PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3",))
    experiment = Experiment(name="exp", run_dir=tmp_path, config=config)

    try:
        load_temporal_data(experiment)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "beit3" in str(exc)


def test_load_temporal_data_uses_frame_id_as_deterministic_tiebreaker(tmp_path) -> None:
    config = PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3",))
    experiment = Experiment(name="exp", run_dir=tmp_path, config=config)
    _write_embeddings(
        tmp_path / "embeddings",
        "beit3",
        ["f2", "f1"],
        np.array([[2.0, 0.0], [1.0, 0.0]], dtype="float32"),
    )

    vectors, records = load_temporal_data(experiment)

    assert [record["frame_id"] for record in records] == ["f1", "f2"]
    assert vectors[:, 0].tolist() == [1.0, 2.0]


def test_load_temporal_data_rejects_vector_id_count_mismatch(tmp_path) -> None:
    config = PipelineConfig(runs_dir=tmp_path, embedding_models=("beit3",))
    experiment = Experiment(name="exp", run_dir=tmp_path, config=config)
    _write_embeddings(
        tmp_path / "embeddings",
        "beit3",
        ["f1"],
        np.array([[1.0, 0.0], [2.0, 0.0]], dtype="float32"),
    )

    with pytest.raises(RetrievalError, match="vectors=2 frame_ids=1"):
        load_temporal_data(experiment)
