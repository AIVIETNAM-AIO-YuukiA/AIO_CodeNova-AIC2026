"""Search orchestration."""

from __future__ import annotations

from core.types import SearchResult
from embedding.clip_model import ClipEmbedder
from index.faiss_index import VectorIndex


class Retriever:
    """Convert text queries to vector searches."""

    def __init__(self, embedder: ClipEmbedder, index: VectorIndex) -> None:
        self.embedder = embedder
        self.index = index

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked frame results for a text query."""
        query_embedding = self.embedder.embed_text(query)
        return self.index.search(query_embedding, top_k=top_k)
