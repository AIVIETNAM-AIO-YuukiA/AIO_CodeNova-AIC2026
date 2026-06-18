"""Full-text index backends (Elasticsearch) for OCR/ASR search."""

from codenova.stores.text.base import TextDocument, TextIndex
from codenova.stores.text.factory import build_text_index

__all__ = ["TextDocument", "TextIndex", "build_text_index"]
