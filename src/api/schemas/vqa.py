"""Request schemas for the VQA / temporal-event domain."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VqaSearchRequest(BaseModel):
    query: str = ""
    question: str = ""
    context: str = ""
    top_k: int | None = Field(default=None, ge=1, le=100)
    reranker_top_k: int | None = Field(default=None, ge=1, le=100)
    vqa_backend: str = "local"
    enabled_models: list[str] | None = None
    use_reranker: bool | None = None
    use_llm: bool = True
    pipeline_mode: Literal["grounded", "legacy"] = "grounded"

    @field_validator("enabled_models")
    @classmethod
    def validate_enabled_models(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = list(dict.fromkeys(model.strip() for model in value if model.strip()))
        if not cleaned:
            raise ValueError("enabled_models must contain at least one model")
        return cleaned


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
