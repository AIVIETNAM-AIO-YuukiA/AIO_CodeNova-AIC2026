"""Request/response schemas for the search domain (/api/search, /api/intelligent-search,
/api/kis-detail-2stage, /api/asr-search, /api/ocr-search).

Field names and defaults mirror the JSON payload shapes ui/api.py's handlers
accepted, so the frontend does not need to change what it sends.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DefaultSearchRequest(BaseModel):
    track: str = "textual_kis"
    query: str = ""
    question: str = ""
    context: str = ""
    top_k: int | None = None
    enabled_models: list[str] | None = None
    use_reranker: bool | None = None
    use_llm: bool | None = None


class IntelligentSearchRequest(BaseModel):
    query: str = ""
    top_k: int | None = None
    enable_kis: bool = True
    enable_ocr: bool = True
    enable_asr: bool = True
    enabled_models: list[str] | None = None
    use_reranker: bool | None = None
    use_llm: bool = True
    fusion_mode: Literal["adaptive", "fixed"] = "adaptive"
    text_search_mode: Literal["separate", "joint"] = "separate"
    temporal_asr: bool = True
    use_evidence_reranker: bool | None = None
    max_frames_per_shot: int = 2


class KisDetail2StageRequest(BaseModel):
    general: list[str] = Field(min_length=1)
    specific: list[str] = Field(min_length=1)
    general_weights: list[float] | None = None
    specific_weights: list[float] | None = None
    enabled_models: list[str] | None = None


class TextSearchRequest(BaseModel):
    query: str = ""
    top_k: int | None = None


class ComputeSubScoreRequest(BaseModel):
    frame_id: str
    sub_text: str
