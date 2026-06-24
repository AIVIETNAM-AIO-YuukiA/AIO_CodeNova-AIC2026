"""On-screen text recognition backends."""

from modules.ocr.base import OcrModel, OcrText
from modules.ocr.factory import build_ocr_model
from modules.ocr.gemini import GeminiOcrModel

__all__ = ["GeminiOcrModel", "OcrModel", "OcrText", "build_ocr_model"]
