"""Online retrieval orchestration."""

from __future__ import annotations

from collections import Counter
import os
import sys
from typing import Protocol

from config.settings import Experiment
from core.logging import get_logger
from core.types import SearchResult
from core.errors import RetrievalError
from indexing.validation import verify_embedding_provenance, verify_frame_files
from modules.embedding import Embedder, build_embedder
from modules.reranker.base import Reranker, build_reranker
from retrieval.fusion import wsf_fuse
import numpy as np
from retrieval.hydrator import ResultHydrator
from retrieval.query_processor import ProcessedQuery, QueryProcessor, get_query_processor
from stores.vector.base import VectorIndex
from stores.vector.factory import build_vector_index

LOGGER = get_logger(__name__)


class RetrievalQuery(Protocol):
    """Structural type accepted by Retriever.search besides a plain string."""

    def to_search_string(self) -> str: ...


class Retriever:
    """Embed a text query, search the index, and hydrate result metadata.

    With a single configured embedding model, behaves as a plain bi-encoder
    retriever. With more than one (e.g. SigLIP + BEiT-3), the query is embedded
    once per model, each model's named vector is searched independently, the
    per-model full-frame scores are combined with frame-ID-aligned WSF
    (see retrieval/fusion.py),
    and the fused candidates are reranked with a cross-encoder (BLIP-2 ITM by
    default) before hydration — mirroring the Cascaded System paper's visual
    search stage: SigLIP + BEiT-3 -> SRRF -> BLIP-2 rerank.

    Construct once (the hydrator caches manifest lookups) and call
    :meth:`search` per query.
    """

    def __init__(
        self,
        experiment: Experiment,
        embedders: dict[str, Embedder],
        index: VectorIndex,
        hydrator: ResultHydrator,
        query_processor: QueryProcessor | None = None,
        reranker: Reranker | None = None,
        fusion_pool_size: int = 100,
    ) -> None:
        if not embedders:
            raise ValueError("Retriever requires at least one embedder.")
        self.experiment = experiment
        self.embedders = embedders
        self.index = index
        self.hydrator = hydrator
        self.query_processor = query_processor or get_query_processor()
        self.reranker = reranker
        self.fusion_pool_size = fusion_pool_size
        self._frame_cache: dict[str, tuple[np.ndarray, list[dict]]] = {}

    def _load_frame_embeddings(self, model_name: str) -> tuple[np.ndarray, list[dict]]:
        if model_name not in self._frame_cache:
            from retrieval.temporal_search import load_temporal_data

            self._frame_cache[model_name] = load_temporal_data(self.experiment, model_name)
        return self._frame_cache[model_name]

    def search(
        self,
        query: str | RetrievalQuery,
        top_k: int = 300,
        enabled_models: list[str] | None = None,
        model_weights: dict[str, float] | None = None,
        use_reranker: bool | None = None,
        use_llm: bool = True,
    ) -> list[SearchResult]:
        """Return ranked, metadata-enriched frame results for a text query.

        ``enabled_models`` restricts the search to a subset of the retriever's
        configured embedders (e.g. from a UI checkbox) without rebuilding
        anything — every embedder is already loaded, this just picks which
        ones vote. ``None`` uses all of them, matching prior behavior.
        ``use_reranker=False`` skips the BLIP-2 rerank step even if one is
        configured; ``None`` defers to whatever the retriever was built with.
        """
        if not isinstance(query, str):
            query = query.to_search_string()

        processed = self.query_processor.process(
            query, enabled_models=enabled_models, use_llm=use_llm
        )
        _log_processed_query(processed)

        return self.search_processed(
            processed,
            top_k=top_k,
            enabled_models=enabled_models,
            model_weights=model_weights,
            use_reranker=use_reranker,
        )

    def search_processed(
        self,
        processed: ProcessedQuery,
        top_k: int = 300,
        enabled_models: list[str] | None = None,
        model_weights: dict[str, float] | None = None,
        use_reranker: bool | None = None,
    ) -> list[SearchResult]:
        """Search using an already decomposed query.

        Intelligent search decomposes a query once for all modalities.  This
        entry point prevents the visual branch from sending the generated
        visual prompt through the LLM a second time.  Plain ``search`` keeps
        its existing public behaviour and delegates here after processing.
        """

        active = self._select_embedders(enabled_models)
        apply_reranker = self.reranker is not None and use_reranker is not False

        pool_size = max(top_k, self.fusion_pool_size) if len(active) > 1 else top_k
        model_data = {}
        for model_name, embedder in active.items():
            if "vietnamese" in model_name.lower() or "vism" in model_name.lower():
                visual_query = processed.visual_prompt_vi
            else:
                visual_query = processed.visual_prompt

            query_embedding = embedder.embed_text(visual_query)
            frame_embs, frame_recs = self._load_frame_embeddings(model_name)
            model_data[model_name] = (frame_embs, frame_recs, query_embedding)

        fused = wsf_fuse(model_data, top_k=pool_size, weights=model_weights)
        hydration = self.hydrator.hydrate_with_diagnostics(fused)
        valid_hydrated = hydration.results
        if hydration.issues:
            reasons = Counter(issue.reason for issue in hydration.issues)
            LOGGER.error(
                "event=RETRIEVAL_CANDIDATES_DROPPED count=%d reasons=%s frame_ids=%s",
                len(hydration.issues),
                dict(sorted(reasons.items())),
                [issue.frame_id for issue in hydration.issues[:20]],
            )

        if apply_reranker:
            rerank_limit = min(50, len(valid_hydrated))
            try:
                reranked_top = self.reranker.rerank(
                    query=processed.visual_prompt, results=valid_hydrated[:rerank_limit]
                )
            except Exception as exc:
                LOGGER.exception(
                    "event=RERANKER_DEGRADED component=retriever-reranker error=%s; "
                    "returning pre-rerank results",
                    exc,
                )
                self.reranker = None
            else:
                valid_hydrated = reranked_top + valid_hydrated[rerank_limit:]
        return valid_hydrated[:top_k]

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
    frame_errors = verify_frame_files(experiment)
    if frame_errors:
        sample = "; ".join(f"{issue['frame_id']}:{issue['reason']}" for issue in frame_errors[:20])
        raise RetrievalError(
            f"Frame artifacts are not safe to serve ({len(frame_errors)} invalid): {sample}"
        )
    provenance_errors = verify_embedding_provenance(experiment)
    if provenance_errors:
        raise RetrievalError(
            "Embedding provenance does not match the offline index: " + "; ".join(provenance_errors)
        )
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
    return Retriever(
        experiment=experiment,
        embedders=embedders,
        index=index,
        hydrator=hydrator,
        reranker=reranker,
    )
