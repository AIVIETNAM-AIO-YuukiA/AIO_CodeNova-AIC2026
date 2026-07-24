"""Vector index build stage (offline)."""

from __future__ import annotations

import json

from config.settings import Experiment
from core.errors import IndexBuildError
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
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install NumPy before building the index.") from exc

    embeddings_dir = experiment.run_dir / "embeddings"
    models = experiment.config.embedding_models
    single_model = len(models) == 1

    per_model_vectors: dict[str, "np.ndarray"] = {}
    per_model_ids: dict[str, list[str]] = {}
    for model_name in models:
        if single_model:
            vectors_path = embeddings_dir / "frames.npz"
            frame_ids_path = embeddings_dir / "frame_ids.json"
        else:
            vectors_path = embeddings_dir / f"frames__{model_name}.npz"
            frame_ids_path = embeddings_dir / f"frame_ids__{model_name}.json"

        if not vectors_path.exists() or not frame_ids_path.exists():
            raise IndexBuildError(
                f"No embeddings found for model '{model_name}' at {vectors_path}. "
                "Run embed-frames first."
            )
        per_model_vectors[model_name] = np.load(vectors_path)["embeddings"].astype("float32")
        per_model_ids[model_name] = json.loads(frame_ids_path.read_text(encoding="utf-8"))

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
