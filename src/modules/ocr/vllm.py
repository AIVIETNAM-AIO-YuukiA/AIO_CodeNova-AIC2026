"""On-screen text extraction over the shared VLM client (see modules/_vllm_chat.py)."""

from __future__ import annotations

import logging

from core.errors import CaptioningError
from modules._vllm_chat import VllmChatClient
from modules.ocr.base import OcrModel
from prompts.ocr import NO_TEXT_MARKER, build_ocr_prompt

LOGGER = logging.getLogger(__name__)

# Greedy decoding: transcription must be deterministic, not creative.
# Measured over sampled news keyframes: the busiest (full ticker + station
# logo + clock + captions) needed 80 tokens, so 200 leaves plenty of headroom.
_GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.1,
    "max_tokens": 200,
    "seed": 42,
}


class VllmOcrModel(OcrModel):
    """Extract on-screen text from keyframes via a self-hosted vLLM endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = VllmChatClient(base_url=base_url, model_name=model_name, timeout=timeout)

    def recognize(self, frame_path: str) -> str:
        """Return on-screen text for one frame image, or "" if none is visible."""
        system_prompt, user_prompt = build_ocr_prompt()
        try:
            text = self._client.complete_with_image(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=frame_path,
                generation_params=_GENERATION_PARAMS,
            )
        except Exception as exc:
            LOGGER.exception("vLLM OCR failed for %s: %s", frame_path, exc)
            raise CaptioningError(f"vLLM OCR failed for {frame_path}: {exc}") from exc

        return "" if text.strip() == NO_TEXT_MARKER else text
