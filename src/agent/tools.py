"""Agent tools — caption/OCR over a frame image, served by the Docker VLM.

Both tools call the vLLM VLM service (``VLLM_BASE_URL``, port 8881 — the same
server the offline captioning/OCR indexing modules use) instead of loading
any model in-process. When that server isn't running they degrade to a
descriptive error string rather than raising, so the agent can still finish a
turn from cached captions and text-index context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)


class Tool(ABC):
    """Base class for all Agent tools."""

    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return result as string."""


class _VlmTool(Tool):
    """Shared plumbing for tools that send one image to the Docker VLM."""

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        self._base_url = base_url
        self._model_name = model_name
        self._client = None

    def _load_client(self):
        if self._client is None:
            from modules._vllm_chat import VllmChatClient

            self._client = VllmChatClient(base_url=self._base_url, model_name=self._model_name)
        return self._client

    def _complete(self, system_prompt: str, user_prompt: str, image_path: str, max_tokens: int):
        if not image_path or not Path(image_path).is_file():
            return f"Error: file not found: {image_path}"
        try:
            return self._load_client().complete_with_image(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=image_path,
                generation_params={"temperature": 0.0, "max_tokens": max_tokens},
            )
        except Exception as exc:
            LOGGER.exception("%s tool failed", self.name)
            return f"[{self.name} unavailable: {exc}. Start the VLM with `make vllm-index`.]"


class CaptionTool(_VlmTool):
    """Describe an image in detail via the Docker VLM."""

    name = "caption"
    description = "Describe an image in detail. Input: image_path. Output: detailed description."

    def run(self, image_path: str = "", prompt: str = "") -> str:
        if not prompt:
            prompt = (
                "Describe this image in detail in Vietnamese. "
                "Focus on: objects, colors, text, people, actions, positions."
            )
        return self._complete(
            system_prompt="You are a precise image captioning assistant.",
            user_prompt=prompt,
            image_path=image_path,
            max_tokens=256,
        )


class OCRTool(_VlmTool):
    """Extract visible text from an image via the Docker VLM."""

    name = "ocr"
    description = "Read all visible text from an image. Input: image_path. Output: extracted text."

    def run(self, image_path: str = "") -> str:
        return self._complete(
            system_prompt="You are a precise OCR assistant.",
            user_prompt="Extract all visible text from this image. Return only the text.",
            image_path=image_path,
            max_tokens=200,
        )


def default_tools() -> dict[str, Tool]:
    """Return the standard VQA tool set (caption + ocr, both Docker-served)."""
    return {"caption": CaptionTool(), "ocr": OCRTool()}
