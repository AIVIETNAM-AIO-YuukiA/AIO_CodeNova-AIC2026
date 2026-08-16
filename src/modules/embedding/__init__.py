"""Image/text embedding models (Jina CLIP v2, SigLIP 2, BEiT-3, Vietnamese captions)."""

from pathlib import Path

from modules.embedding.base import Embedder
from modules.embedding.beit3 import Beit3Embedder
from modules.embedding.jina import JinaClipEmbedder
from modules.embedding.siglip import SiglipEmbedder
from modules.embedding.vietnamese import VietnameseEmbedder
from modules.embedding.registry import (
    EmbeddingModelSpec,
    resolve_embedding_model,
    resolve_embedding_models,
)


def build_embedder(
    model_name: str,
    device: str = "auto",
    batch_size: int = 32,
    captions_path: Path | None = None,
) -> Embedder:
    """Build the strictly resolved backend for ``model_name``."""
    spec = resolve_embedding_model(model_name)
    if spec.backend == "JinaClipEmbedder":
        return JinaClipEmbedder(
            model_name=model_name if "/" in model_name else None,
            device=device,
            batch_size=batch_size,
        )
    if spec.backend == "Beit3Embedder":
        return Beit3Embedder(model_name=model_name, device=device, batch_size=batch_size)
    if spec.backend == "VietnameseEmbedder":
        return VietnameseEmbedder(
            model_name=model_name if "/" in model_name else None,
            device=device,
            batch_size=batch_size,
            captions_path=captions_path,
        )
    if spec.backend == "SiglipEmbedder":
        return SiglipEmbedder(
            model_name=model_name if "/" in model_name else None,
            device=device,
            batch_size=batch_size,
        )
    raise AssertionError(f"Unhandled embedding backend: {spec.backend}")


__all__ = [
    "Beit3Embedder",
    "Embedder",
    "EmbeddingModelSpec",
    "JinaClipEmbedder",
    "SiglipEmbedder",
    "VietnameseEmbedder",
    "build_embedder",
    "resolve_embedding_model",
    "resolve_embedding_models",
]
