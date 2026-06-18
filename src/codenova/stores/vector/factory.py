"""Build a vector index from experiment configuration."""

from __future__ import annotations

import os

from codenova.config.settings import Experiment
from codenova.core.errors import IndexBuildError
from codenova.stores.vector.base import VectorIndex
from codenova.stores.vector.qdrant import QdrantVectorIndex


def build_vector_index(experiment: Experiment) -> VectorIndex:
    """Create the configured vector index for an experiment.

    Qdrant connection details come from the environment (loaded from the root
    ``.env`` at startup). The per-experiment collection name is namespaced with
    the experiment name so separate runs never collide in a shared instance.
    """
    backend = experiment.config.index_backend
    if backend != "qdrant":
        raise IndexBuildError(f"Unsupported index backend '{backend}'. Only 'qdrant' is supported.")

    base_collection = os.environ.get("QDRANT_COLLECTION", "codenova_frames")
    return QdrantVectorIndex(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection=f"{base_collection}__{experiment.name}",
        api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
