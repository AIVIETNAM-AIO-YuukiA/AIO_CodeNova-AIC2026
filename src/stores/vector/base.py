"""Vector index interface shared by all backends."""

from __future__ import annotations

from core.types import SearchResult


class VectorIndex:
    """Interface for vector index implementations."""

    def build(self, embeddings: list[list[float]], frame_ids: list[str]) -> None:
        """Build (or upsert into) an index from embeddings and their frame IDs."""
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Return the top-k nearest frame IDs with similarity scores."""
        raise NotImplementedError


def frame_result(frame_id: str, score: float) -> SearchResult:
    """Create a minimal index search result (metadata hydrated downstream)."""
    return SearchResult(frame_id=frame_id, video_id="", score=score)
