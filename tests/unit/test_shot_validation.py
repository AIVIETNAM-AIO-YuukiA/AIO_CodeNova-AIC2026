"""Tests for gather_frame_s and ShotValidator."""

import numpy as np

from retrieval.temporal_search import (
    ShotInput,
    ShotValidator,
    gather_frame_s,
)


def test_gather_frame_s_basic() -> None:
    segment = {"start_pos": 5, "end_pos": 8, "length": 4, "center_idx": 6}
    records = [
        {
            "frame_id": f"f{i:03d}",
            "video_id": "v1",
            "timestamp_sec": i * 0.5,
            "frame_path": f"frames/v1/f{i:03d}.jpg",
        }
        for i in range(15)
    ]

    shot = gather_frame_s(segment, records)
    assert shot is not None
    assert shot.video_id == "v1"
    assert shot.frame_count == 4


def test_gather_frame_s_none_result() -> None:
    records = [{"frame_id": "f001", "video_id": "v1"}]
    shot = gather_frame_s(None, records)
    assert shot is None


def test_gather_frame_s_boundary_start() -> None:
    segment = {"start_pos": 0, "end_pos": 2, "length": 3, "center_idx": 1}
    records = [
        {"frame_id": f"f{i:03d}", "video_id": "v1", "frame_path": f"frames/v1/f{i:03d}.jpg"}
        for i in range(5)
    ]

    shot = gather_frame_s(segment, records)
    assert shot is not None
    assert shot.frame_count == 3


def test_shot_validator_good_shot() -> None:
    rng = np.random.RandomState(42)
    n_frames = 20

    base = rng.randn(64).astype("float32")
    query = base / np.linalg.norm(base)

    similar = np.array([base + rng.randn(64).astype("float32") * 0.05 for _ in range(10)])
    similar = similar / np.linalg.norm(similar, axis=1, keepdims=True)
    different = rng.randn(10, 64).astype("float32")
    different = different / np.linalg.norm(different, axis=1, keepdims=True)
    all_embeddings = np.concatenate([similar, different], axis=0)

    records = [
        {"frame_id": f"f{i:03d}", "video_id": "v1", "timestamp_sec": i * 0.5}
        for i in range(n_frames)
    ]

    shot = ShotInput(
        video_id="v1",
        video_name="",
        frames=records[:5],
        frame_paths=[f"frames/v1/f{i:03d}.jpg" for i in range(5)],
        frame_count=5,
    )

    validator = ShotValidator(min_frames=2, min_sim_score=0.1)
    result = validator.validate(shot, query, all_embeddings, records)

    assert result.validated is True
    assert result.validation_score > 0


def test_shot_validator_too_few_frames() -> None:
    rng = np.random.RandomState(42)
    query = rng.randn(64).astype("float32")
    all_embeddings = rng.randn(10, 64).astype("float32")
    records = [{"frame_id": f"f{i:03d}", "video_id": "v1"} for i in range(10)]

    shot = ShotInput(
        video_id="v1",
        video_name="",
        frames=records[:1],
        frame_paths=["frames/v1/f000.jpg"],
        frame_count=1,
    )

    validator = ShotValidator(min_frames=2)
    result = validator.validate(shot, query, all_embeddings, records)

    assert result.validated is False
    assert result.validation_score == 0.0


def test_shot_validator_estimates_temporal_consistency() -> None:
    rng = np.random.RandomState(42)
    query = rng.randn(64).astype("float32")
    query = query / np.linalg.norm(query)

    all_embeddings = np.tile(query, (5, 1)).astype("float32")
    records = [
        {"frame_id": f"f{i:03d}", "video_id": "v1", "timestamp_sec": float(i)} for i in range(5)
    ]

    shot = ShotInput(
        video_id="v1",
        video_name="",
        frames=records[:3],
        frame_paths=[f"frames/v1/f{i:03d}.jpg" for i in range(3)],
        frame_count=3,
    )

    validator = ShotValidator()
    result = validator.validate(shot, query, all_embeddings, records)

    assert result.temporal_score > 0
    assert result.validated is True
