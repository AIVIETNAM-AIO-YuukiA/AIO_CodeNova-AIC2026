"""Request schemas for the VQA / temporal-event domain."""

from __future__ import annotations

from pydantic import BaseModel


class VqaSearchRequest(BaseModel):
    query: str = ""
    question: str = ""
    context: str = ""
    top_k: int | None = None
    reranker_top_k: int | None = None
    vqa_backend: str = "local"


class TrakeOrEnhancedSearchRequest(BaseModel):
    query: str = ""
    context: str = ""
    events: list[str] | list[dict] | None = None
    top_k: int | None = None
    window: int = 15
    max_events: int = 5
    enabled_models: list[str] | None = None
    use_reranker: bool | None = None
    use_llm: bool | None = None
    reranker_top_k: int | None = None
