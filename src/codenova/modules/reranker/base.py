"""Reranker interface.

Stub only: defines the contract so a BLIP-2 / cross-encoder reranker can be
added later without touching the retrieval layer.
"""

from __future__ import annotations

from codenova.core.types import SearchResult


class Reranker:
    """Interface for second-stage rerankers (BLIP-2 ITM, cross-encoder)."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Reorder candidate results for a query, returning the new ranking."""
        raise NotImplementedError
