"""File naming for per-model embedding output — the single source of truth.

Every model always gets its own ``frames__<model>.npz`` /
``frame_ids__<model>.json``, even when only one model is configured, so two
embed-frames runs for different models never collide on the same file.
"""

from __future__ import annotations

from pathlib import Path


def vectors_path(embeddings_dir: Path, model_name: str) -> Path:
    return embeddings_dir / f"frames__{model_name}.npz"


def frame_ids_path(embeddings_dir: Path, model_name: str) -> Path:
    return embeddings_dir / f"frame_ids__{model_name}.json"


def checkpoint_vectors_path(embeddings_dir: Path, model_name: str) -> Path:
    """Path to a model's in-progress (not yet finalized) vectors file."""
    return vectors_path(embeddings_dir, model_name).with_suffix(".checkpoint.npz")


def checkpoint_frame_ids_path(embeddings_dir: Path, model_name: str) -> Path:
    """Path to a model's in-progress (not yet finalized) frame-ids file."""
    return frame_ids_path(embeddings_dir, model_name).with_suffix(".checkpoint.json")
