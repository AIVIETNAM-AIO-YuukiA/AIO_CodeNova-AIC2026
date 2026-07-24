"""OCR interface.

See ``modules/ocr/vllm.py`` for the concrete backend (self-hosted vLLM,
same server as captioning but a dedicated text-only-extraction prompt).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrText:
    """Text recognized on a single frame."""

    frame_id: str
    video_id: str
    text: str


class OcrModel:
    """Interface for OCR backends (PaddleOCR, Gemini Flash)."""

    def recognize(self, frame_path: str) -> str:
        """Return on-screen text for one frame image."""
        raise NotImplementedError
