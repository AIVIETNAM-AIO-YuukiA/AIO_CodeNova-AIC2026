"""Conversational agent domain routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_experiment
from api.schemas.agent import AgentChatRequest
from api.services import agent_service

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/chat")
def chat(req: AgentChatRequest, experiment=Depends(get_experiment)):
    try:
        return agent_service.run_agent_chat(experiment, req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
