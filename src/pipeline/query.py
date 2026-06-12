"""Query orchestration placeholder."""

from __future__ import annotations

from core.types import SearchResult
from retrieval.search import Retriever


def run_query(retriever: Retriever, query: str, top_k: int) -> list[SearchResult]:
    """Run a text query against a prepared retrieval index."""
    return retriever.search(query=query, top_k=top_k)
