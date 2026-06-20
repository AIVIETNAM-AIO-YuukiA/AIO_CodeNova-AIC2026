"""Image/text embedding models (SigLIP 2)."""

from modules.embedding.base import Embedder
from modules.embedding.siglip import SiglipEmbedder


def build_embedder(model_name: str, device: str = "auto", batch_size: int = 32) -> Embedder:
    """Return the embedder for ``model_name`` (currently SigLIP 2 only)."""
    return SiglipEmbedder(model_name=model_name, device=device, batch_size=batch_size)


__all__ = ["Embedder", "SiglipEmbedder", "build_embedder"]
