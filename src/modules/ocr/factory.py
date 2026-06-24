"""Build OCR backends from environment configuration."""

from __future__ import annotations

import os

from core.errors import CodeNovaError
from modules.ocr.base import OcrModel
from modules.ocr.gemini import GeminiOcrModel


def build_ocr_model() -> OcrModel:
    """Create the configured OCR backend."""
    backend = os.environ.get("OCR_BACKEND", "gemini")
    if backend == "gemini":
        return GeminiOcrModel(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            model_name=os.environ.get("OCR_GEMINI_MODEL", "gemini-3.5-flash"),
            inline_max_mb=int(os.environ.get("OCR_INLINE_MAX_MB", "20")),
        )
    raise CodeNovaError(f"Unsupported OCR_BACKEND '{backend}'. Only 'gemini' is supported.")
