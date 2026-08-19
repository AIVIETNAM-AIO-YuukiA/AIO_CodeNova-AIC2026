"""Text index interface.

Full-text (BM25) search over OCR / ASR text. CLIP text encoders cap at ~77
tokens, so a lexical text index complements vector search for long queries and
exact matches (names, signs, dialogue). Documents share the same ``frame_id`` /
``video_id`` keys as the vector store so results can be fused.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextDocument:
    """One indexable text document tied to a frame or video."""

    doc_id: str
    video_id: str
    text: str
    source: str  # "ocr" | "asr" | "caption"
    frame_id: str | None = None
    timestamp_sec: float | None = None


class TextIndex:
    """Interface for full-text index backends (Elasticsearch)."""

    def index_documents(self, documents: list[TextDocument]) -> None:
        """Index a batch of text documents."""
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` pairs for a BM25 text query."""
        raise NotImplementedError

    def search_documents(
        self,
        query: str,
        top_k: int,
        source: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[dict]:
        """Return full matching documents (text + metadata + ``score``).

        ``source`` optionally restricts to one or several modalities.
        Unlike ``search`` (ids only, for fusion), this returns the document
        bodies — used by the interactive agent, which needs to read the text.
        """
        raise NotImplementedError

    def export_all(self):
        """Yield every indexed document as a dict, for local backup/inspection."""
        raise NotImplementedError
