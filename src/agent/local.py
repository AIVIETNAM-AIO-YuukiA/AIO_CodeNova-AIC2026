"""Local model inference via Ollama for VQA tools and reasoning.

Requires: ``ollama`` Python package (``uv add ollama``), Ollama server
installed from https://ollama.com/download, model pulled
(e.g. ``ollama pull llava``), and server running (``ollama serve``).
"""

from __future__ import annotations

from pathlib import Path
import json
import logging
import os
import re

import ollama

from agent.brain import BrainResponse
from agent.tools import Tool

LOGGER = logging.getLogger(__name__)

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=_OLLAMA_HOST)


def is_ollama_running() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        _ollama_client().list()
        return True
    except Exception:
        return False


def _list_local_models() -> list[str]:
    """Return list of model names available locally via Ollama."""
    try:
        resp = _ollama_client().list()
        return [m["name"] for m in resp.get("models", [])]
    except Exception:
        return []


def _pick_vision_model() -> str:
    """Pick the best available vision model. Forces moondream when available."""
    models = _list_local_models()
    forced = os.environ.get("OLLAMA_VISION_MODEL", "")
    if forced:
        return forced
    if "moondream" in str(models).lower():
        return "moondream:latest"
    for name in ["moondream", "llama3.2-vision", "llava:7b", "llava", "qwen2.5-vl", "minicpm-v"]:
        for m in models:
            if m.startswith(name):
                return m
    return "moondream:latest"


# ---------------------------------------------------------------------------
# Tools (caption, ocr)  —  use vision model
# ---------------------------------------------------------------------------


class LocalCaptionTool(Tool):
    """Describe an image using a local Ollama vision model."""

    name = "caption"
    description = (
        "Describe an image in detail. Input: image_path. "
        "Output: a detailed English description of what is visible."
    )

    def __init__(self, model_name: str | None = None) -> None:
        self.client = _ollama_client()
        self.model_name = model_name or _pick_vision_model()
        LOGGER.info("LocalCaptionTool using model=%s", self.model_name)

    def run(self, image_path: str = "", prompt: str = "Describe this image in detail.") -> str:
        if not image_path:
            return "Error: image_path is required."
        if not Path(image_path).is_file():
            return f"Error: file not found: {image_path}"
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_path],
                options={"num_predict": 512},
            )
            return response.get("response", "").strip()
        except Exception as exc:
            LOGGER.exception("LocalCaptionTool failed")
            return f"[Caption error: {exc}]"


class LocalOCRTool(Tool):
    """Read text from an image using a local Ollama vision model."""

    name = "ocr"
    description = (
        "Read and extract text visible in an image. Input: image_path. "
        "Output: all text found in the image."
    )

    def __init__(self, model_name: str | None = None) -> None:
        self.client = _ollama_client()
        self.model_name = model_name or _pick_vision_model()
        LOGGER.info("LocalOCRTool using model=%s", self.model_name)

    def run(self, image_path: str = "") -> str:
        if not image_path:
            return "Error: image_path is required."
        if not Path(image_path).is_file():
            return f"Error: file not found: {image_path}"
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt="Extract all visible text from this image. Return only the text found.",
                images=[image_path],
                options={"num_predict": 512},
            )
            return response.get("response", "").strip()
        except Exception as exc:
            LOGGER.exception("LocalOCRTool failed")
            return f"[OCR error: {exc}]"


# ---------------------------------------------------------------------------
# Brain (text reasoning)  —  can use any Ollama model
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a VQA (Video Question Answering) agent.
Your task is to answer questions about video frames accurately.

You have access to these tools:
1. caption(image_path): Describe the image in detail (objects, actions, colors, people).
2. ocr(image_path): Extract all visible written text, words, numbers, and signs from the image.

Rules:
- Read the question carefully. 
- If the question asks for names, exact text, poetry, or numbers -> YOU MUST use the `ocr` tool.
- If the question asks for visual descriptions, colors, or actions -> YOU MUST use the `caption` tool.
- Input format for tools: {"image_path": "file.jpg"}
- Answer the user directly after receiving the tool's observation. Keep it short and precise.

Examples of Tool Calling:
{"thought": "The user asks for the name on the sign. I need to read the text.", "action": "ocr", "action_input": {"image_path": "file.jpg"}}

Example of Answering:
{"thought": "The OCR result says 'Nguyen Trung Truc'. That is the answer.", "answer": "Nguyễn Trung Trực", "finished": true}
"""


class LocalBrain:
    """Local model brain powered by Ollama for ReAct reasoning.

    Args:
        model_name: Ollama model name (default: auto-pick vision model).
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.client = _ollama_client()
        self.model_name = model_name or _pick_vision_model()
        LOGGER.info("LocalBrain using model=%s", self.model_name)

    def reset(self) -> None:
        pass

    def reason(
        self,
        question: str,
        shot_info: str,
        frame_count: int,
        tool_results: list[dict] | None = None,
    ) -> BrainResponse:
        """Send context + question to the local LLM and parse structured response."""
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

        prompt = f"{SYSTEM_PROMPT}\n\n{context}"

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"num_predict": 512},
                format="json",
            )
            return self._parse_response(response.get("response", "").strip())
        except Exception as exc:
            LOGGER.exception("LocalBrain reason failed")
            return BrainResponse(answer=f"Reasoning error: {exc}", finished=True)

    def _parse_response(self, text: str) -> BrainResponse:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            LOGGER.warning("No JSON in local brain response: %s", text[:200])
            return BrainResponse(answer=text.strip(), finished=True)
        try:
            data = json.loads(json_match.group())
            if data.get("finished") or data.get("answer"):
                return BrainResponse(
                    thought=data.get("thought", ""),
                    answer=data.get("answer", ""),
                    finished=True,
                )
            return BrainResponse(
                thought=data.get("thought", ""),
                action=data.get("action", ""),
                action_input=data.get("action_input", {}),
                finished=False,
            )
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse local brain JSON: %s", exc)
            return BrainResponse(answer=text.strip(), finished=True)


def local_default_tools() -> dict[str, Tool]:
    """Return dict of local Ollama-based tools."""
    vision_model = _pick_vision_model()
    return {
        "caption": LocalCaptionTool(model_name=vision_model),
        "ocr": LocalOCRTool(model_name=vision_model),
        "detect": __import__("agent.tools", fromlist=["DetectTool"]).DetectTool(),
        "asr": __import__("agent.tools", fromlist=["ASRTool"]).ASRTool(),
    }
