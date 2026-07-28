"""Vector index build stage (offline)."""

from __future__ import annotations

import json

from config.settings import Experiment
from core.errors import IndexBuildError
from indexing.embedding_paths import (
    checkpoint_frame_ids_path,
    checkpoint_vectors_path,
    frame_ids_path,
    vectors_path,
)
from stores.vector.factory import build_vector_index


def build_index(experiment: Experiment, force: bool = False) -> int:
    """Upsert saved frame embeddings (all configured models) into the vector index.

    Returns the number of points indexed. ``force`` is accepted for CLI parity;
    the index backend always recreates its collection on build, so a rebuild is
    inherently idempotent.

    If multiple embedding models are configured but have embedded different
    numbers of frames (e.g. one model's ``embed-frames`` run is behind), only
    the frames common to every model are indexed, so every point has a
    complete set of named vectors.

    Falls back to a model's in-progress checkpoint (``*.checkpoint.npz`` /
    ``*.checkpoint.json``) when the finished file doesn't exist yet — lets you
    build and search a partial index while ``embed-frames`` is still running,
    to sanity-check results before the full run completes. The checkpoint
    writer flushes atomically (temp file + rename), so this is always safe to
    read even mid-write.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install NumPy before building the index.") from exc

    embeddings_dir = experiment.run_dir / "embeddings"
    models = experiment.config.embedding_models

    per_model_vectors: dict[str, "np.ndarray"] = {}
    per_model_ids: dict[str, list[str]] = {}
    for model_name in models:
        model_vectors_path = vectors_path(embeddings_dir, model_name)
        model_frame_ids_path = frame_ids_path(embeddings_dir, model_name)

        if not model_vectors_path.exists() or not model_frame_ids_path.exists():
            model_vectors_path = checkpoint_vectors_path(embeddings_dir, model_name)
            model_frame_ids_path = checkpoint_frame_ids_path(embeddings_dir, model_name)

        if not model_vectors_path.exists() or not model_frame_ids_path.exists():
            raise IndexBuildError(
                f"No embeddings (finished or in-progress) found for model '{model_name}' "
                f"under {embeddings_dir}. Run embed-frames first."
            )
        per_model_vectors[model_name] = np.load(model_vectors_path)["embeddings"].astype("float32")
        per_model_ids[model_name] = json.loads(model_frame_ids_path.read_text(encoding="utf-8"))

    # Frames common to every model, in the first model's order.
    common_ids = set(per_model_ids[models[0]])
    for model_name in models[1:]:
        common_ids &= set(per_model_ids[model_name])
    frame_ids = [fid for fid in per_model_ids[models[0]] if fid in common_ids]
    if not frame_ids:
        raise IndexBuildError("No frames are embedded by every configured model.")

    embeddings_by_model: dict[str, list[list[float]]] = {}
    for model_name in models:
        id_to_row = {fid: row for row, fid in enumerate(per_model_ids[model_name])}
        rows = [id_to_row[fid] for fid in frame_ids]
        embeddings_by_model[model_name] = per_model_vectors[model_name][rows].tolist()

    index = build_vector_index(experiment)
    index.build(embeddings_by_model, frame_ids)
    return len(frame_ids)
