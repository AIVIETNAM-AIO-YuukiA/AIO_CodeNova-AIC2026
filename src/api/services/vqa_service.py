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


def _attach_frame_image_urls(result: dict) -> dict:
    for video in result.get("videos", []):
        for ev in video.get("events", []):
            if ev.get("frame_path"):
                ev["image_url"] = f"/frame?path={quote(ev['frame_path'])}"
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    for ev in result.get("events", []):
        ev["image_urls"] = [f"/frame?path={quote(fp)}" for fp in ev.get("frame_paths", []) if fp]
    return result


def run_vqa_search(
    experiment: Experiment,
    default_top_k: int,
    reranker,
    reranker_top_k: int,
    req: VqaSearchRequest,
) -> dict:
    result = vqa_search(
        experiment=experiment,
        query=req.query,
        question=req.question,
        context=req.context,
        top_k=req.top_k or default_top_k,
        reranker=reranker if req.reranker_top_k else None,
        reranker_top_k=req.reranker_top_k or reranker_top_k,
        vqa_backend=req.vqa_backend,
    )
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    return result


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
