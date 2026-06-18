"""Online retrieval orchestration."""

from __future__ import annotations

from config.settings import Experiment
from core.types import SearchResult
from stores.vector.base import VectorIndex
from stores.vector.factory import build_vector_index
from modules.embedding import ClipEmbedder, TransformersClipEmbedder
from retrieval.hydrator import ResultHydrator


class Retriever:
    """Embed a text query, search the index, and hydrate result metadata.

    Construct once (the hydrator caches manifest lookups) and call
    :meth:`search` per query.
    """

    def __init__(
        self,
        embedder: ClipEmbedder,
        index: VectorIndex,
        hydrator: ResultHydrator,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.hydrator = hydrator

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked, metadata-enriched frame results for a text query."""
        query_embedding = self.embedder.embed_text(query)
        raw_results = self.index.search(query_embedding, top_k=top_k)
        return self.hydrator.hydrate(raw_results)


def build_retriever(experiment: Experiment) -> Retriever:
    """Assemble a Retriever from an experiment's configuration."""
    embedder = TransformersClipEmbedder(
        model_name=experiment.config.clip_model,
        device=experiment.config.device,
    )
    index = build_vector_index(experiment)
    hydrator = ResultHydrator(experiment)
    return Retriever(embedder=embedder, index=index, hydrator=hydrator)
