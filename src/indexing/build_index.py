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

    Every configured model must contain the same unique frame-ID set. Row order
    may differ because vectors are joined explicitly by frame ID before build.

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
    if not models:
        raise IndexBuildError("At least one embedding model is required.")

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
        vectors = np.load(model_vectors_path)["embeddings"].astype("float32")
        ids = json.loads(model_frame_ids_path.read_text(encoding="utf-8"))
        if vectors.ndim != 2 or not isinstance(ids, list):
            raise IndexBuildError(
                f"Invalid embedding artifact model={model_name} shape={vectors.shape}"
            )
        ids = [str(value) for value in ids]
        if len(vectors) != len(ids):
            raise IndexBuildError(
                f"Embedding vector/ID mismatch model={model_name} "
                f"vectors={len(vectors)} ids={len(ids)}"
            )
        if len(ids) != len(set(ids)):
            raise IndexBuildError(f"Duplicate frame IDs for model={model_name}")
        if not np.isfinite(vectors).all():
            raise IndexBuildError(f"Non-finite embedding vectors for model={model_name}")
        per_model_vectors[model_name] = vectors
        per_model_ids[model_name] = ids

    # Canonical frame order comes from the first model; every other model is
    # joined to it by frame ID rather than assumed to share row positions.
    frame_ids = per_model_ids[models[0]]
    canonical_ids = set(frame_ids)
    for model_name in models[1:]:
        model_ids = set(per_model_ids[model_name])
        if model_ids != canonical_ids:
            missing = sorted(canonical_ids - model_ids)
            extra = sorted(model_ids - canonical_ids)
            raise IndexBuildError(
                f"Embedding frame-ID set mismatch model={model_name} "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
    if not frame_ids:
        raise IndexBuildError("Embedding artifacts contain no frame IDs.")

    # Keep vectors as numpy arrays (not `.tolist()`) — converting ~300k rows to
    # nested Python lists inflates RAM 6-8x (each float becomes its own object).
    embeddings_by_model: dict[str, "np.ndarray"] = {}
    for model_name in models:
        id_to_row = {fid: row for row, fid in enumerate(per_model_ids[model_name])}
        rows = [id_to_row[fid] for fid in frame_ids]
        embeddings_by_model[model_name] = per_model_vectors[model_name][rows]

    index = build_vector_index(experiment)
    index.build(embeddings_by_model, frame_ids)
    return len(frame_ids)
