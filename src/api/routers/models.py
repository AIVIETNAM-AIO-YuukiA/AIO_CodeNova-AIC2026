"""Model-discovery route — lets the UI render its model checkboxes/dropdowns
from what's actually configured for this experiment, instead of a hardcoded
list baked into HTML/JS. Add a model to an experiment's embedding_models (or
change RERANKER_BACKEND) and it shows up here with no frontend change.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends

from api.deps import get_reranker, get_retriever
from api.schemas.models import ModelsResponse

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def list_models(retriever=Depends(get_retriever), reranker=Depends(get_reranker)):
    return ModelsResponse(
        embedding_models=list(retriever.embedders),
        reranker_backend=os.environ.get("RERANKER_BACKEND", "blip2"),
        reranker_available=reranker is not None,
        llm_model=os.environ.get("OPENROUTER_MODEL_FOR_CHAT") or os.environ.get("OPENROUTER_MODEL"),
    )
