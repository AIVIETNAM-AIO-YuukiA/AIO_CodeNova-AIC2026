"""Online retrieval orchestration."""

from __future__ import annotations

import sys
from config.settings import Experiment
from core.logging import get_logger
from core.types import SearchResult
from stores.vector.base import VectorIndex
from stores.vector.factory import build_vector_index
from modules.embedding import Embedder, build_embedder
from retrieval.hydrator import ResultHydrator
from retrieval.query_processor import QueryProcessor, get_query_processor

LOGGER = get_logger(__name__)


class Retriever:
    """Embed a text query, search the index, and hydrate result metadata.

    Construct once (the hydrator caches manifest lookups) and call
    :meth:`search` per query.
    """

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        hydrator: ResultHydrator,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.hydrator = hydrator
        self.query_processor = query_processor or get_query_processor()

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked, metadata-enriched frame results for a text query."""
        # Translate and enhance query via LLM if configured, otherwise pass-through
        processed = self.query_processor.process(query)

        LOGGER.info("Query processing completed:")
        LOGGER.info("  - Raw query: %s", processed.raw_query)
        LOGGER.info("  - Visual prompt: %s", processed.visual_prompt)
        if processed.ocr_keywords:
            LOGGER.info("  - OCR keywords: %s", processed.ocr_keywords)
        if processed.asr_keywords:
            LOGGER.info("  - ASR keywords: %s", processed.asr_keywords)
        if processed.metadata:
            LOGGER.info("  - Metadata: %s", processed.metadata)

        print("\n--- [Query Processor] ---", file=sys.stderr)
        print(f"  Raw query: {processed.raw_query}", file=sys.stderr)
        print(f"  Visual prompt:  {processed.visual_prompt}", file=sys.stderr)
        if processed.ocr_keywords:
            print(f"  OCR keywords: {processed.ocr_keywords}", file=sys.stderr)
        if processed.asr_keywords:
            print(f"  ASR keywords: {processed.asr_keywords}", file=sys.stderr)
        if processed.metadata:
            print(f"  Metadata: {processed.metadata}", file=sys.stderr)
        print("-------------------------\n", file=sys.stderr)

        query_embedding = self.embedder.embed_text(processed.visual_prompt)
        raw_results = self.index.search(query_embedding, top_k=top_k)
        return self.hydrator.hydrate(raw_results)


def build_retriever(experiment: Experiment) -> Retriever:
    """Assemble a Retriever from an experiment's configuration."""
    embedder = build_embedder(
        model_name=experiment.config.embedding_model,
        device=experiment.config.device,
    )
    index = build_vector_index(experiment)
    hydrator = ResultHydrator(experiment)
    return Retriever(embedder=embedder, index=index, hydrator=hydrator)

