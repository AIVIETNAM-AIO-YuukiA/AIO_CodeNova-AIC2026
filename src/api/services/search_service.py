"""Search domain service — bridges validated request schemas to the existing
retrieval logic (retrieval/search.py, retrieval/intelligent_search.py,
retrieval/kis_detail_search.py, retrieval/text_search.py) and shapes the
response for the frontend (image_url, video_name).

Routers call these functions with typed request objects; nothing here knows
about HTTP status codes or raw JSON dicts.
"""

from __future__ import annotations

from urllib.parse import quote

import numpy as np

from api.schemas.search import (
    ComputeSubScoreRequest,
    DefaultSearchRequest,
    IntelligentSearchRequest,
    KisDetail2StageRequest,
    TextSearchRequest,
)
from config.settings import Experiment
from core.errors import RetrievalError
from retrieval.intelligent_search import intelligent_search
from retrieval.kis_detail_search import kis_detail_2stage_search
from retrieval.text_search import text_search
from retrieval.tracks import SUPPORTED_TRACKS, TrackQuery, build_retrieval_text
from ui.api import result_to_payload

DEFAULT_TOP_K = 20


def _attach_image_urls(result: dict) -> dict:
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    return result


def run_default_search(
    retriever, experiment: Experiment, default_top_k: int, req: DefaultSearchRequest
) -> dict:
    request = TrackQuery(
        track=req.track, query=req.query, question=req.question, context=req.context
    )
    retrieval_text = build_retrieval_text(request)
    results = retriever.search(
        query=retrieval_text,
        top_k=req.top_k or default_top_k,
        enabled_models=req.enabled_models,
        use_reranker=req.use_reranker,
        use_llm=req.use_llm if req.use_llm is not None else True,
    )
    return {
        "track": request.track,
        "track_label": SUPPORTED_TRACKS.get(request.track, request.track),
        "retrieval_text": retrieval_text,
        "results": [result_to_payload(result, experiment) for result in results],
    }


def run_intelligent_search(
    experiment: Experiment, default_top_k: int, req: IntelligentSearchRequest
) -> dict:
    result = intelligent_search(
        experiment,
        query=req.query,
        top_k=req.top_k or default_top_k,
        enable_kis=req.enable_kis,
        enable_ocr=req.enable_ocr,
        enable_asr=req.enable_asr,
        enabled_models=req.enabled_models,
        use_reranker=req.use_reranker,
        use_llm=req.use_llm,
        fusion_mode=req.fusion_mode,
        text_search_mode=req.text_search_mode,
        temporal_asr=req.temporal_asr,
        use_evidence_reranker=req.use_evidence_reranker,
        max_frames_per_shot=req.max_frames_per_shot,
    )
    return _attach_image_urls(result)


def run_kis_detail_2stage(experiment: Experiment, req: KisDetail2StageRequest) -> dict:
    general = [s.strip() for s in req.general if s.strip()]
    specific = [s.strip() for s in req.specific if s.strip()]
    if not general or not specific:
        raise ValueError("At least 1 non-empty general and specific subquery are required.")

    result = kis_detail_2stage_search(
        experiment=experiment,
        general=general,
        specific=specific,
        general_weights=req.general_weights,
        specific_weights=req.specific_weights,
        enabled_models=req.enabled_models,
    )
    return _attach_image_urls(result)


def run_text_search(
    experiment: Experiment, default_top_k: int, source: str, req: TextSearchRequest
) -> dict:
    result = text_search(experiment, query=req.query, source=source, top_k=req.top_k or default_top_k)
    return _attach_image_urls(result)


def compute_sub_score(retriever, req: ComputeSubScoreRequest) -> float:
    model_name = next(iter(retriever.embedders))
    frame_vec = retriever.index.get_vector(req.frame_id, model_name)
    if frame_vec is None:
        raise RetrievalError(f"frame_id not found: {req.frame_id}")

    frame_vec = np.asarray(frame_vec, dtype="float32")
    sub_vec = np.asarray(
        retriever.embedders[model_name].embed_text(req.sub_text), dtype="float32"
    ).flatten()
    norm = np.linalg.norm(sub_vec)
    if norm > 1e-12:
        sub_vec /= norm
    return round(float(frame_vec @ sub_vec), 4)
