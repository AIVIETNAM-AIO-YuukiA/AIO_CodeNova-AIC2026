"""Online retrieval orchestration."""

from __future__ import annotations

from collections import Counter
import os
import sys
from time import perf_counter
from typing import Protocol

from config.settings import Experiment
from core.logging import get_logger
from core.types import SearchResult
from core.errors import RetrievalError
from indexing.validation import verify_embedding_provenance, verify_frame_files
from modules.embedding import Embedder, build_embedder
from modules.reranker.base import Reranker, build_reranker
from retrieval.fusion import srrf_fuse
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

        tick = perf_counter()
        processed = self.query_processor.process(
            query, enabled_models=enabled_models, use_llm=use_llm
        )
        LOGGER.info("event=SEARCH_TIMING timing_ms={'query_processing': %s}", _elapsed_ms(tick))
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
        use_expansion: bool = True,
        num_expansions: int = 4,
    ) -> list[SearchResult]:
        """Search using an already decomposed query.

        Intelligent search decomposes a query once for all modalities.  This
        entry point prevents the visual branch from sending the generated
        visual prompt through the LLM a second time.  Plain ``search`` keeps
        its existing public behaviour and delegates here after processing.

        ``use_expansion`` mirrors the AIC_2025 reference project's
        ``auto_expand``: the LLM generates extra visually-descriptive query
        variants, each searched independently per model, with all variants'
        candidates pooled together before fusion — widening recall for vague
        or single-phrasing queries. Disabled automatically when unavailable
        (LLM circuit open) or explicitly turned off by the caller.
        """

        started = perf_counter()
        timing_ms: dict[str, float] = {}

        active = self._select_embedders(enabled_models)
        apply_reranker = self.reranker is not None and use_reranker is not False

        expansions: list[str] = []
        if use_expansion:
            tick = perf_counter()
            expansions = self.query_processor.expand_query(
                processed.visual_prompt, num_expansions=num_expansions
            )
            timing_ms["expand_query"] = _elapsed_ms(tick)

        pool_size = max(top_k, self.fusion_pool_size) if len(active) > 1 else top_k
        results_by_model = {}
        for model_name, embedder in active.items():
            if "vietnamese" in model_name.lower() or "vism" in model_name.lower():
                visual_queries = [processed.visual_prompt_vi]
            else:
                visual_queries = [processed.visual_prompt, *expansions]

            tick = perf_counter()
            best_by_frame_id: dict[str, SearchResult] = {}
            for visual_query in visual_queries:
                query_embedding = embedder.embed_text(visual_query)
                for result in self.index.search(
                    query_embedding, top_k=pool_size, model_name=model_name
                ):
                    # A frame_id found by more than one query variant keeps
                    # its best score, not a summed one — SRRF fusion expects
                    # one entry per frame_id per model.
                    existing = best_by_frame_id.get(result.frame_id)
                    if existing is None or result.score > existing.score:
                        best_by_frame_id[result.frame_id] = result
            timing_ms[f"embed_search:{model_name}"] = _elapsed_ms(tick)
            results_by_model[model_name] = list(best_by_frame_id.values())

        tick = perf_counter()
        fused = srrf_fuse(results_by_model, top_k=pool_size, weights=model_weights)
        timing_ms["fusion"] = _elapsed_ms(tick)

        tick = perf_counter()
        hydration = self.hydrator.hydrate_with_diagnostics(fused)
        timing_ms["hydration"] = _elapsed_ms(tick)
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
            rerank_limit = min(100, len(valid_hydrated))
            tick = perf_counter()
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
            timing_ms["rerank"] = _elapsed_ms(tick)

        timing_ms["total"] = _elapsed_ms(started)
        LOGGER.info("event=SEARCH_TIMING timing_ms=%s", timing_ms)
        return valid_hydrated[:top_k]

    def search_variant_branches(
        self,
        query_variants: dict[str, list[str] | tuple[str, ...]],
        *,
        top_k: int = 500,
        enabled_models: list[str] | None = None,
    ) -> dict[str, list[SearchResult]]:
        """Search explicit bilingual variants without cross-model fusion.

        Grounded VQA needs high recall before it can assemble an ordered
        moment.  SRRF is still appropriate for the normal KIS result list,
        but it can push a frame that is strong in only one model out of the
        candidate pool.  This additive API exposes every model/variant branch
        independently so the VQA layer can union hits at video level first.

        ``query_variants`` is keyed by a descriptive language/source label
        (normally ``"en"`` and ``"vi"``).  The label is included in the
        returned branch key for diagnostics; all non-empty variants are sent
        to every enabled embedder because both Jina CLIP v2 and SigLIP2 can
        contribute useful multilingual hits in practice.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")

        active = self._select_embedders(enabled_models)
        branches: dict[str, list[SearchResult]] = {}
        for model_name, embedder in active.items():
            for variant_group, raw_variants in query_variants.items():
                seen_queries: set[str] = set()
                for variant_index, raw_query in enumerate(raw_variants):
                    query = str(raw_query).strip()
                    if not query or query in seen_queries:
                        continue
                    seen_queries.add(query)
                    query_embedding = embedder.embed_text(query)
                    raw_results = self.index.search(
                        query_embedding,
                        top_k=top_k,
                        model_name=model_name,
                    )
                    hydration = self.hydrator.hydrate_with_diagnostics(raw_results)
                    if hydration.issues:
                        reasons = Counter(issue.reason for issue in hydration.issues)
                        LOGGER.error(
                            "event=RETRIEVAL_CANDIDATES_DROPPED branch=%s:%s:%d "
                            "count=%d reasons=%s frame_ids=%s",
                            model_name,
                            variant_group,
                            variant_index,
                            len(hydration.issues),
                            dict(sorted(reasons.items())),
                            [issue.frame_id for issue in hydration.issues[:20]],
                        )
                    branches[
                        f"{model_name}:{variant_group}:{variant_index}"
                    ] = hydration.results[:top_k]
        return branches

    def _select_embedders(self, enabled_models: list[str] | None) -> dict[str, Embedder]:
        if enabled_models is None:
            return self.embedders
        active = {name: emb for name, emb in self.embedders.items() if name in enabled_models}
        if not active:
            raise ValueError(
                f"None of {enabled_models} match configured models {list(self.embedders)}."
            )
        return active


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


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
    configured, also builds a reranker to score fused SRRF candidates —
    BLIP-2 ITM by default, or Qwen3-VL via vLLM when RERANKER_BACKEND=qwen-vl-vllm
    (see modules/reranker/base.py). Set ``DISABLE_RERANKER=1`` to skip it
    entirely, which frees the VRAM/HTTP round-trip it costs so several
    embedders fit on a small GPU.
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
