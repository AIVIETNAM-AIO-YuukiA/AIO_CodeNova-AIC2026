"""Online retrieval orchestration."""

from __future__ import annotations

import sys

from config.settings import Experiment
from core.logging import get_logger
from core.types import SearchResult
from modules.embedding import Embedder, build_embedder
from modules.reranker.base import Reranker, build_reranker
from retrieval.fusion import srrf_fuse
from retrieval.hydrator import ResultHydrator
from retrieval.query_processor import QueryProcessor, get_query_processor
from stores.vector.base import VectorIndex
from stores.vector.factory import build_vector_index

LOGGER = get_logger(__name__)


class Retriever:
    """Embed a text query, search the index, and hydrate result metadata.

    With a single configured embedding model, behaves as a plain bi-encoder
    retriever. With more than one (e.g. SigLIP + BEiT-3), the query is embedded
    once per model, each model's named vector is searched independently, the
    per-model result lists are combined with SRRF (see retrieval/fusion.py),
    and the fused candidates are reranked with a cross-encoder (BLIP-2 ITM by
    default) before hydration — mirroring the Cascaded System paper's visual
    search stage: SigLIP + BEiT-3 -> SRRF -> BLIP-2 rerank.

    Construct once (the hydrator caches manifest lookups) and call
    :meth:`search` per query.
    """

    def __init__(
        self,
        embedders: dict[str, Embedder],
        index: VectorIndex,
        hydrator: ResultHydrator,
        query_processor: QueryProcessor | None = None,
        reranker: Reranker | None = None,
        fusion_pool_size: int = 100,
    ) -> None:
        if not embedders:
            raise ValueError("Retriever requires at least one embedder.")
        self.embedders = embedders
        self.index = index
        self.hydrator = hydrator
        self.query_processor = query_processor or get_query_processor()
        self.reranker = reranker
        self.fusion_pool_size = fusion_pool_size

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked, metadata-enriched frame results for a text query."""
        processed = self.query_processor.process(query)
        _log_processed_query(processed)

        if len(self.embedders) == 1:
            ((model_name, embedder),) = self.embedders.items()
            query_embedding = embedder.embed_text(processed.visual_prompt)
            raw_results = self.index.search(query_embedding, top_k=top_k, model_name=model_name)
            return self.hydrator.hydrate(raw_results)

        pool_size = max(top_k, self.fusion_pool_size)
        per_model_results = []
        for model_name, embedder in self.embedders.items():
            query_embedding = embedder.embed_text(processed.visual_prompt)
            per_model_results.append(
                self.index.search(query_embedding, top_k=pool_size, model_name=model_name)
            )

        fused = srrf_fuse(per_model_results, top_k=pool_size)
        if self.reranker is not None:
            fused = self.reranker.rerank(query=processed.visual_prompt, results=fused)
        return self.hydrator.hydrate(fused[:top_k])


def _log_processed_query(processed) -> None:
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


def build_retriever(experiment: Experiment) -> Retriever:
    """Assemble a Retriever from an experiment's configuration.

    Builds one embedder per configured model. When more than one model is
    configured, also builds the default reranker (BLIP-2 ITM) to score fused
    SRRF candidates.
    """
    embedders = {
        model_name: build_embedder(model_name=model_name, device=experiment.config.device)
        for model_name in experiment.config.embedding_models
    }
    index = build_vector_index(experiment)
    hydrator = ResultHydrator(experiment)
    reranker = (
        build_reranker("blip2-itm", device=experiment.config.device) if len(embedders) > 1 else None
    )
    return Retriever(embedders=embedders, index=index, hydrator=hydrator, reranker=reranker)
