"""Search domain routes — HTTP shape only, no business logic here.

Each route validates the request through its Pydantic schema, calls exactly
one service function, and returns its result. Model selection is a request
field (``enabled_models``), never a separate route per model — see
routers/models.py for how the frontend discovers what's available.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_default_top_k, get_experiment, get_retriever
from api.schemas.search import (
    ComputeSubScoreRequest,
    DefaultSearchRequest,
    IntelligentSearchRequest,
    KisDetail2StageRequest,
    TextSearchRequest,
)
from api.services import search_service
from core.errors import RetrievalError

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search")
def search(
    req: DefaultSearchRequest,
    retriever=Depends(get_retriever),
    experiment=Depends(get_experiment),
    default_top_k=Depends(get_default_top_k),
):
    try:
        return search_service.run_default_search(retriever, experiment, default_top_k, req)
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/intelligent-search")
def intelligent_search(
    req: IntelligentSearchRequest,
    experiment=Depends(get_experiment),
    default_top_k=Depends(get_default_top_k),
):
    try:
        return search_service.run_intelligent_search(experiment, default_top_k, req)
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/kis-detail-2stage")
def kis_detail_2stage(req: KisDetail2StageRequest, experiment=Depends(get_experiment)):
    try:
        return search_service.run_kis_detail_2stage(experiment, req)
    except (ValueError, RetrievalError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/asr-search")
def asr_search(
    req: TextSearchRequest, experiment=Depends(get_experiment), default_top_k=Depends(get_default_top_k)
):
    return search_service.run_text_search(experiment, default_top_k, "asr", req)


@router.post("/ocr-search")
def ocr_search(
    req: TextSearchRequest, experiment=Depends(get_experiment), default_top_k=Depends(get_default_top_k)
):
    return search_service.run_text_search(experiment, default_top_k, "ocr", req)


@router.post("/compute-sub-score")
def compute_sub_score(req: ComputeSubScoreRequest, retriever=Depends(get_retriever)):
    try:
        score = search_service.compute_sub_score(retriever, req)
    except RetrievalError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"score": score}
