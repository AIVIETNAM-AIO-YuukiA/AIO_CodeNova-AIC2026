"""On-screen text recognition (OCR) modules."""

from modules.ocr.base import OcrModel, OcrText
from modules.ocr.vllm import VllmOcrModel

__all__ = ["OcrModel", "OcrText", "VllmOcrModel"]
