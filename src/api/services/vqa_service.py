"""VQA / temporal-event domain service — bridges request schemas to
retrieval/vqa.py and retrieval/trake_search.py, and shapes the response
(image_url on every frame reference).
"""

from __future__ import annotations

from urllib.parse import quote

from api.schemas.vqa import TrakeOrEnhancedSearchRequest, VqaSearchRequest
from config.settings import Experiment
from retrieval.vqa import enhanced_temporal_search, trake_search, vqa_search
from ui.api import events_to_query


def _attach_image_urls(value: object) -> None:
    """Attach browser-safe frame URLs to every nested frame reference.

    Grounded VQA adds frame references below ``evidence_frames``,
    ``selected_candidate`` and ``candidate_answers``.  Walking the JSON-like
    response keeps those additions backward compatible with the existing
    top-level ``results``/``events`` response shapes.
    """
    if isinstance(value, list):
        for item in value:
            _attach_image_urls(item)
        return
    if not isinstance(value, dict):
        return

    frame_path = value.get("frame_path")
    if isinstance(frame_path, str) and frame_path:
        value.setdefault("image_url", f"/frame?path={quote(frame_path)}")

    frame_paths = value.get("frame_paths")
    if isinstance(frame_paths, list):
        value.setdefault(
            "image_urls",
            [f"/frame?path={quote(fp)}" for fp in frame_paths if isinstance(fp, str) and fp],
        )

    for nested in value.values():
        _attach_image_urls(nested)


def _attach_frame_image_urls(result: dict) -> dict:
    _attach_image_urls(result)
    return result


def run_vqa_search(
    experiment: Experiment,
    default_top_k: int,
    reranker,
    reranker_top_k: int,
    req: VqaSearchRequest,
) -> dict:
    reranker_requested = (
        req.use_reranker if req.use_reranker is not None else req.reranker_top_k is not None
    )
    result = vqa_search(
        experiment=experiment,
        query=req.query,
        question=req.question,
        context=req.context,
        top_k=req.top_k or default_top_k,
        reranker=reranker if reranker_requested else None,
        reranker_top_k=req.reranker_top_k or reranker_top_k,
        vqa_backend=req.vqa_backend,
        enabled_models=req.enabled_models,
        use_reranker=req.use_reranker,
        use_llm=req.use_llm,
        pipeline_mode=req.pipeline_mode,
    )
    return _attach_frame_image_urls(result)


def run_trake_search(
    experiment: Experiment,
    default_top_k: int,
    reranker,
    reranker_top_k: int,
    req: TrakeOrEnhancedSearchRequest,
) -> dict:
    """TRAKE search — bidirectional pair-join for 2+ explicit events, else the
    legacy single-string-query path (which itself falls back to BPJ if the
    query text parses into 2+ ``E1:``/``E2:``-style events).
    """
    if isinstance(req.events, list) and len(req.events) >= 2:
        from retrieval.trake_search import trake_bpj_search

        result = trake_bpj_search(
            experiment=experiment,
            events=req.events,
            top_k=300,
            window=req.window,
            enabled_models=req.enabled_models,
            use_reranker=req.use_reranker,
            use_llm=req.use_llm,
        )
    else:
        query = events_to_query({"events": req.events, "query": req.query})
        result = trake_search(
            experiment=experiment,
            query=query,
            context=req.context,
            top_k=req.top_k or default_top_k,
            reranker=reranker if req.reranker_top_k else None,
            reranker_top_k=req.reranker_top_k or reranker_top_k,
        )
    return _attach_frame_image_urls(result)


def run_enhanced_temporal_search(
    experiment: Experiment,
    default_top_k: int,
    reranker,
    reranker_top_k: int,
    req: TrakeOrEnhancedSearchRequest,
) -> dict:
    result = enhanced_temporal_search(
        experiment=experiment,
        query=req.query,
        context=req.context,
        max_events=req.max_events,
        top_k=req.top_k or default_top_k,
        reranker=reranker if req.reranker_top_k else None,
        reranker_top_k=req.reranker_top_k or reranker_top_k,
    )
    return _attach_frame_image_urls(result)
