"""Vector index interface shared by all backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.types import SearchResult

if TYPE_CHECKING:
    import numpy as np


class VectorIndex:
    """Interface for vector index implementations.

    Supports one or more named embedding spaces per point (e.g. "siglip" and
    "beit3"). Single-model callers just use one key in ``embeddings_by_model``.
    """

    def build(self, embeddings_by_model: dict[str, "np.ndarray"], frame_ids: list[str]) -> None:
        """Build (or upsert into) an index from named embeddings and frame IDs.

        ``embeddings_by_model`` maps a model name to its 2D float32 array of
        vectors (row-aligned with ``frame_ids``). Arrays, not Python lists, so
        callers pass through the joined embeddings without an expensive
        numpy-to-list conversion of every element.
        """
        raise NotImplementedError

    def search(
        self, query_embedding: list[float], top_k: int, model_name: str | None = None
    ) -> list[SearchResult]:
        """Return the top-k nearest frame IDs with similarity scores.

        ``model_name`` selects which named vector to search when the index
        holds more than one; required for multi-model indexes.
        """
        raise NotImplementedError

    def get_vector(self, frame_id: str, model_name: str) -> list[float] | None:
        """Return the stored vector for one frame, or ``None`` if not found.

        Point lookup by payload filter, not a full-corpus scan — for reading
        a single frame's embedding without loading every vector into RAM.
        """
        raise NotImplementedError


def frame_result(frame_id: str, score: float) -> SearchResult:
    """Create a minimal index search result (metadata hydrated downstream)."""
    return SearchResult(frame_id=frame_id, video_id="", score=score)
