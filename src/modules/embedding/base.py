"""Embedding interface and shared helpers for all backends."""

from __future__ import annotations

from core.errors import EmbeddingError
from core.types import FrameRecord


class Embedder:
    """Interface for image/text embedding backends."""

    def embed_images(self, frames: list[FrameRecord]) -> list[list[float]]:
        """Embed image frames."""
        raise NotImplementedError

    def embed_text(self, query: str) -> list[float]:
        """Embed a text query."""
        raise NotImplementedError


def projected_features(features):
    """Return the projected embedding tensor across Transformers versions.

    Transformers 4.x returned tensors from ``get_image_features`` and
    ``get_text_features``. Transformers 5.x returns ``BaseModelOutputWithPooling``
    and stores the projected embedding in ``pooler_output``.
    """
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def resolve_torch_device(torch, requested: str):
    """Resolve a torch device and require CUDA for auto/cuda modes."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise EmbeddingError("CUDA is not available; embedding requires a GPU in this project.")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise EmbeddingError(f"Requested device '{requested}' but CUDA is not available.")
    return torch.device(requested)
