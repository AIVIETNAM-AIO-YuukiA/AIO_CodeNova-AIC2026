"""Conversational agent domain service — bridges to agent/interactive.py."""

from __future__ import annotations

from urllib.parse import quote

from api.schemas.agent import AgentChatRequest
from config.settings import Experiment


def run_agent_chat(experiment: Experiment, req: AgentChatRequest) -> dict:
    from agent.interactive import run_agent_turn

    result = run_agent_turn(req.messages, experiment)
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    return result
