"""Vector index build stage (offline)."""

from __future__ import annotations

import json

from config.settings import Experiment
from index.factory import build_vector_index


def build_index(experiment: Experiment, force: bool = False) -> int:
    """Upsert saved frame embeddings into the configured vector index.

    Returns the number of embeddings indexed. ``force`` is accepted for CLI
    parity; the index backend always recreates its collection on build, so a
    rebuild is inherently idempotent.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install NumPy before building the index.") from exc

    embeddings_dir = experiment.run_dir / "embeddings"
    vectors_path = embeddings_dir / "frames.npz"
    frame_ids_path = embeddings_dir / "frame_ids.json"

    vectors = np.load(vectors_path)["embeddings"].astype("float32")
    frame_ids = json.loads(frame_ids_path.read_text(encoding="utf-8"))

    index = build_vector_index(experiment)
    index.build(vectors.tolist(), frame_ids)
    return len(frame_ids)
