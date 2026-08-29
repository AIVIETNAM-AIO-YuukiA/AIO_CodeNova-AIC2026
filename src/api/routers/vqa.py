"""VQA / temporal-event domain routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_default_top_k, get_experiment, get_reranker, get_reranker_top_k
from api.schemas.vqa import TrakeOrEnhancedSearchRequest, VqaSearchRequest
from api.services import vqa_service

router = APIRouter(prefix="/api", tags=["vqa"])


@router.post("/vqa-search")
def vqa_search(
    req: VqaSearchRequest,
    experiment=Depends(get_experiment),
    default_top_k=Depends(get_default_top_k),
    reranker=Depends(get_reranker),
    reranker_top_k=Depends(get_reranker_top_k),
):
    try:
        return vqa_service.run_vqa_search(experiment, default_top_k, reranker, reranker_top_k, req)
    except Exception as exc:  # ported from the do_POST handler's broad catch
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/trake-search")
def trake_search(
    req: TrakeOrEnhancedSearchRequest,
    experiment=Depends(get_experiment),
    default_top_k=Depends(get_default_top_k),
    reranker=Depends(get_reranker),
    reranker_top_k=Depends(get_reranker_top_k),
):
    try:
        return vqa_service.run_trake_search(experiment, default_top_k, reranker, reranker_top_k, req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/enhanced-temporal-search")
def enhanced_temporal_search(
    req: TrakeOrEnhancedSearchRequest,
    experiment=Depends(get_experiment),
    default_top_k=Depends(get_default_top_k),
    reranker=Depends(get_reranker),
    reranker_top_k=Depends(get_reranker_top_k),
):
    try:
        return vqa_service.run_enhanced_temporal_search(
            experiment, default_top_k, reranker, reranker_top_k, req
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
