"""Response schema for /api/models — lets the UI render model checkboxes/dropdowns
from what's actually configured, instead of a hardcoded list in HTML/JS.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelsResponse(BaseModel):
    embedding_models: list[str]
    reranker_backend: str
    reranker_available: bool
    llm_model: str | None
