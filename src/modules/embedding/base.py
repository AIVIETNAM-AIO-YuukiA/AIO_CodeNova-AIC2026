"""Embedding interface shared by all backends."""

from __future__ import annotations

from core.types import FrameRecord


class ClipEmbedder:
    """Interface for CLIP-style image/text embeddings."""

    def embed_images(self, frames: list[FrameRecord]) -> list[list[float]]:
        """Embed image frames."""
        raise NotImplementedError

    def embed_text(self, query: str) -> list[float]:
        """Embed a text query."""
        raise NotImplementedError
