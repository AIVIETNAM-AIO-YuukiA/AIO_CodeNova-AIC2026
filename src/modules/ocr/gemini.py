"""Gemini-backed OCR implementation."""

from __future__ import annotations

from pathlib import Path
import mimetypes

from core.errors import CodeNovaError
from modules.ocr.base import OcrModel

OCR_PROMPT = (
    "Extract all visible text from this video frame. Return only the text. "
    "Preserve useful line breaks. If no readable text is visible, return an empty string."
)


class GeminiOcrModel(OcrModel):
    """OCR backend using Gemini image understanding."""

    def __init__(self, api_key: str, model_name: str, inline_max_mb: int = 20) -> None:
        if not api_key:
            raise CodeNovaError("GEMINI_API_KEY is required for OCR_BACKEND=gemini.")
        self.api_key = api_key
        self.model_name = model_name
        self.inline_max_bytes = inline_max_mb * 1024 * 1024
        self._client = None

    def recognize(self, frame_path: str) -> str:
        """Return on-screen text for one frame image."""
        path = Path(frame_path)
        if not path.exists():
            raise CodeNovaError(f"OCR frame does not exist: {frame_path}")

        client = self._load_client()
        if path.stat().st_size <= self.inline_max_bytes:
            response = client.models.generate_content(
                model=self.model_name,
                contents=[OCR_PROMPT, _inline_image_part(path)],
            )
        else:
            uploaded = client.files.upload(file=path)
            response = client.models.generate_content(
                model=self.model_name,
                contents=[OCR_PROMPT, uploaded],
            )
        return (getattr(response, "text", "") or "").strip()

    def _load_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise CodeNovaError("Install google-genai before using Gemini OCR.") from exc
        self._client = genai.Client(api_key=self.api_key)
        return self._client


def _inline_image_part(path: Path) -> dict[str, object]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": path.read_bytes(),
        }
    }
