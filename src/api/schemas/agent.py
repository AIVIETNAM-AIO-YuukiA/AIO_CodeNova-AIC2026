"""Request schemas for the conversational agent domain."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(min_length=1)
