"""Agent brain — the Docker-hosted LLM behind every agent code path.

One backend only: an OpenAI-compatible ``/v1/chat/completions`` server,
not served by this repo's ``docker-compose.yml`` — point ``.env``'s
``AGENT_LOCAL_ENGINE_URL`` at whatever OpenAI-compatible endpoint you run
for it. The brain reasons over text (the served Qwen3.5-4B is text-only);
anything visual reaches it through the caption/ocr tools in
``agent/tools.py``, which call the separate vLLM VLM service, or through
captions cached at index time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import re

from prompts.agent import VQA_SYSTEM_PROMPT

LOGGER = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8884/v1"
_DEFAULT_MODEL = "cyankiwi/Qwen3.5-4B-AWQ-4bit"


@dataclass
class BrainResponse:
    """Structured response from the agent brain."""

    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    answer: str = ""
    finished: bool = False


class AgentBrain:
    """ReAct reasoning via the Docker-hosted agent LLM (OpenAI-compatible)."""

    def __init__(self, model_name: str | None = None, base_url: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("AGENT_LOCAL_ENGINE_URL", _DEFAULT_BASE_URL)
        self.model_name = (
            model_name or os.environ.get("AGENT_LOCAL_ENGINE_MODEL") or _DEFAULT_MODEL
        )
        self._client = None

    def reset(self) -> None:
        """No per-conversation state to clear; kept for the Agent loop's protocol."""

    def reason(
        self,
        question: str,
        shot_info: str,
        frame_count: int,
        tool_results: list[dict] | None = None,
    ) -> BrainResponse:
        """Send context + question to the LLM and parse the JSON ReAct response."""
        context = f"Question: {question}\nShot info: {shot_info}\nFrames available: {frame_count}\n"
        if tool_results:
            history = "\n".join(
                f"Tool {r.get('tool', '?')} returned: {r.get('result', '')}" for r in tool_results
            )
            context += f"\nPrevious observations:\n{history}"

        try:
            client = self._load_client()
            text = client.complete_text(
                system_prompt=VQA_SYSTEM_PROMPT,
                user_prompt=context,
                generation_params={"temperature": 0.0, "max_tokens": 512},
            )
            return parse_brain_response(text)
        except Exception as exc:
            LOGGER.exception("Brain reasoning failed")
            return BrainResponse(
                answer=f"Agent LLM unavailable ({exc}). Check AGENT_LOCAL_ENGINE_URL in .env.",
                finished=True,
            )

    def _load_client(self):
        if self._client is not None:
            return self._client
        from modules._vllm_chat import VllmChatClient

        self._client = VllmChatClient(base_url=self.base_url, model_name=self.model_name)
        return self._client


def parse_brain_response(text: str) -> BrainResponse:
    """Parse the ``{"thought"/"action"/"answer"}`` JSON the ReAct prompt mandates.

    Non-JSON output is treated as a final answer rather than an error — small
    local models occasionally drop the format on the last step, and the text
    itself is usually the answer.
    """
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return BrainResponse(answer=text.strip(), finished=True)
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return BrainResponse(answer=text.strip(), finished=True)
    if data.get("finished") or data.get("answer"):
        return BrainResponse(
            thought=data.get("thought", ""), answer=data.get("answer", ""), finished=True
        )
    return BrainResponse(
        thought=data.get("thought", ""),
        action=data.get("action", ""),
        action_input=data.get("action_input", {}),
        finished=False,
    )
