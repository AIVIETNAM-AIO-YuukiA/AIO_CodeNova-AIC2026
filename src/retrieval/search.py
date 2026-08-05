"""Online retrieval orchestration."""

from __future__ import annotations

import os
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

    def search(
        self,
        query: str,
        top_k: int,
        enabled_models: list[str] | None = None,
        model_weights: dict[str, float] | None = None,
        use_reranker: bool | None = None,
    ) -> list[SearchResult]:
        """Return ranked, metadata-enriched frame results for a text query.

        ``enabled_models`` restricts the search to a subset of the retriever's
        configured embedders (e.g. from a UI checkbox) without rebuilding
        anything — every embedder is already loaded, this just picks which
        ones vote. ``None`` uses all of them, matching prior behavior.
        ``use_reranker=False`` skips the BLIP-2 rerank step even if one is
        configured; ``None`` defers to whatever the retriever was built with.
        """
        processed = self.query_processor.process(query)
        _log_processed_query(processed)

        active = self._select_embedders(enabled_models)
        apply_reranker = self.reranker is not None and use_reranker is not False

        if len(active) == 1:
            ((model_name, embedder),) = active.items()
            query_embedding = embedder.embed_text(processed.visual_prompt)
            raw_results = self.index.search(query_embedding, top_k=top_k, model_name=model_name)
            # Hydrate before reranking: the index returns frame ids only, and
            # the cross-encoder needs each result's frame_path to load its image.
            hydrated = self.hydrator.hydrate(raw_results)
            if apply_reranker:
                hydrated = self.reranker.rerank(query=processed.visual_prompt, results=hydrated)
            return hydrated[:top_k]

        pool_size = max(top_k, self.fusion_pool_size)
        results_by_model = {}
        for model_name, embedder in active.items():
            query_embedding = embedder.embed_text(processed.visual_prompt)
            results_by_model[model_name] = self.index.search(
                query_embedding, top_k=pool_size, model_name=model_name
            )

        fused = srrf_fuse(results_by_model, top_k=pool_size, weights=model_weights)
        hydrated = self.hydrator.hydrate(fused)
        if apply_reranker:
            hydrated = self.reranker.rerank(query=processed.visual_prompt, results=hydrated)
        return hydrated[:top_k]

    def _select_embedders(self, enabled_models: list[str] | None) -> dict[str, Embedder]:
        if enabled_models is None:
            return self.embedders
        active = {name: emb for name, emb in self.embedders.items() if name in enabled_models}
        if not active:
            raise ValueError(
                f"None of {enabled_models} match configured models {list(self.embedders)}."
            )
        return active



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
    SRRF candidates — set ``DISABLE_RERANKER=1`` to skip it, which frees the
    ~1.5 GB of VRAM it holds so several embedders fit on a small GPU.
    """
    embedders = {
        model_name: build_embedder(model_name=model_name, device=experiment.config.device)
        for model_name in experiment.config.embedding_models
    }
    index = build_vector_index(experiment)
    hydrator = ResultHydrator(experiment)
    disable_reranker = os.environ.get("DISABLE_RERANKER", "0").lower() not in ("0", "false", "")
    reranker = (
        build_reranker("blip2-itm", device=experiment.config.device)
        if len(embedders) > 1 and not disable_reranker
        else None
    )
    return Retriever(embedders=embedders, index=index, hydrator=hydrator, reranker=reranker)
