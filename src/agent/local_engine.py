"""Self-hosted local-engine brain (Atlas or llama.cpp) for the VQA/TRAKE agent.

Fallback tier below Gemini in create_agent() (react.py): when no
GEMINI_API_KEY is set, the agent talks to whichever local engine
docker-compose.yml started (atlas-agent or llamacpp-agent — see that file's
comments for why there are two). Both expose an OpenAI-compatible
/v1/chat/completions endpoint on port 8888, so this reuses the same
VllmChatClient the offline indexing modules use for vllm-index/-retrieval —
just pointed at a different port and model.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from agent.brain import SYSTEM_PROMPT, BrainResponse
from agent.tools import Tool

LOGGER = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8888/v1"


class LocalEngineBrain:
    """Brain backed by a self-hosted Atlas or llama.cpp server (OpenAI-compatible API)."""

    def __init__(self, model_name: str | None = None, base_url: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("AGENT_LOCAL_ENGINE_URL", _DEFAULT_BASE_URL)
        self.model_name = model_name or os.environ.get("AGENT_LOCAL_ENGINE_MODEL", "")
        self._client = None
        self._frame_paths: list[str] = []

    def set_frame_paths(self, frame_paths: list[str]) -> None:
        """Set candidate frame paths so reason() has an image to send."""
        self._frame_paths = frame_paths

    def reset(self) -> None:
        self._frame_paths = []

    def reason(
        self,
        question: str,
        shot_info: str,
        frame_count: int,
        tool_results: list[dict] | None = None,
    ) -> BrainResponse:
        """Send the shot's centre frame + question to the local engine, deterministic (temp=0)."""
        image_path = self._frame_paths[len(self._frame_paths) // 2] if self._frame_paths else ""
        if not image_path:
            return BrainResponse(
                answer="No frame available for local-engine reasoning.", finished=True
            )

        context = (
            f"Question: {question}\n"
            f"Shot info: {shot_info}\n"
            f"Frames available: {frame_count}\n"
        )
        if tool_results:
            history = "\n".join(
                f"Tool {r.get('tool', '?')} returned: {r.get('result', '')}" for r in tool_results
            )
            context += f"\nPrevious observations:\n{history}"

        try:
            client = self._load_client()
            text = client.complete_with_image(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=context,
                image_path=image_path,
                generation_params={"temperature": 0.0, "max_tokens": 512},
            )
            return _parse_response(text)
        except Exception as exc:
            LOGGER.exception("Local-engine brain reasoning failed")
            return BrainResponse(answer=f"Local-engine reasoning error: {exc}", finished=True)

    def _load_client(self):
        if self._client is not None:
            return self._client
        from modules._vllm_chat import VllmChatClient

        self._client = VllmChatClient(base_url=self.base_url, model_name=self.model_name)
        return self._client


def _parse_response(text: str) -> BrainResponse:
    """Parse the same {"thought"/"action"/"answer"} JSON format brain.py expects."""
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return BrainResponse(answer=text.strip(), finished=True)
    try:
        data = json.loads(json_match.group())
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
    except json.JSONDecodeError:
        return BrainResponse(answer=text.strip(), finished=True)


class LocalEngineCaptionTool(Tool):
    """Describe an image using the local engine (Atlas or llama.cpp)."""

    name = "caption"
    description = "Describe an image in detail. Input: image_path. Output: detailed description."

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("AGENT_LOCAL_ENGINE_URL", _DEFAULT_BASE_URL)
        self.model_name = model_name or os.environ.get("AGENT_LOCAL_ENGINE_MODEL", "")
        self._client = None

    def run(self, image_path: str = "", prompt: str = "") -> str:
        if not image_path or not Path(image_path).is_file():
            return f"Error: file not found: {image_path}"
        if not prompt:
            prompt = (
                "Describe this image in detail in Vietnamese. "
                "Focus on: objects, colors, text, people, actions, positions."
            )
        try:
            client = self._load_client()
            return client.complete_with_image(
                system_prompt="You are a precise image captioning assistant.",
                user_prompt=prompt,
                image_path=image_path,
                generation_params={"temperature": 0.0, "max_tokens": 256},
            )
        except Exception as exc:
            LOGGER.exception("LocalEngineCaptionTool failed")
            return f"[Caption error: {exc}]"

    def _load_client(self):
        if self._client is None:
            from modules._vllm_chat import VllmChatClient

            self._client = VllmChatClient(base_url=self.base_url, model_name=self.model_name)
        return self._client


class LocalEngineOCRTool(Tool):
    """Extract visible text from an image using the local engine."""

    name = "ocr"
    description = "Read all visible text from an image. Input: image_path. Output: extracted text."

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("AGENT_LOCAL_ENGINE_URL", _DEFAULT_BASE_URL)
        self.model_name = model_name or os.environ.get("AGENT_LOCAL_ENGINE_MODEL", "")
        self._client = None

    def run(self, image_path: str = "") -> str:
        if not image_path or not Path(image_path).is_file():
            return f"Error: file not found: {image_path}"
        try:
            client = self._load_client()
            return client.complete_with_image(
                system_prompt="You are a precise OCR assistant.",
                user_prompt="Extract all visible text from this image. Return only the text.",
                image_path=image_path,
                generation_params={"temperature": 0.0, "max_tokens": 200},
            )
        except Exception as exc:
            LOGGER.exception("LocalEngineOCRTool failed")
            return f"[OCR error: {exc}]"

    def _load_client(self):
        if self._client is None:
            from modules._vllm_chat import VllmChatClient

            self._client = VllmChatClient(base_url=self.base_url, model_name=self.model_name)
        return self._client


def local_engine_default_tools(
    base_url: str | None = None, model_name: str | None = None
) -> dict[str, Tool]:
    """Return the local engine's own caption/ocr tools plus the Gemini-independent
    detect/asr tools — none of these need GEMINI_API_KEY, matching the reason
    this brain exists (it's the no-cloud-key fallback)."""
    from agent.tools import ASRTool, DetectTool

    return {
        "caption": LocalEngineCaptionTool(base_url=base_url, model_name=model_name),
        "ocr": LocalEngineOCRTool(base_url=base_url, model_name=model_name),
        "detect": DetectTool(),
        "asr": ASRTool(),
    }
