"""Full-text index backends (Elasticsearch) for OCR/ASR search."""

from stores.text.base import TextDocument, TextIndex
from stores.text.factory import build_text_index

__all__ = ["TextDocument", "TextIndex", "build_text_index"]
