"""Vector index interface shared by all backends."""

from __future__ import annotations

from core.types import SearchResult


class VectorIndex:
    """Interface for vector index implementations.

    Supports one or more named embedding spaces per point (e.g. "siglip" and
    "beit3"). Single-model callers just use one key in ``embeddings_by_model``.
    """

    def build(
        self, embeddings_by_model: dict[str, list[list[float]]], frame_ids: list[str]
    ) -> None:
        """Build (or upsert into) an index from named embeddings and frame IDs.

        ``embeddings_by_model`` maps a model name to its list of vectors; every
        model's vector list must have the same length and order as ``frame_ids``.
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


def frame_result(frame_id: str, score: float) -> SearchResult:
    """Create a minimal index search result (metadata hydrated downstream)."""
    return SearchResult(frame_id=frame_id, video_id="", score=score)
