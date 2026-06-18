"""Image/text embedding models."""

from modules.embedding.base import ClipEmbedder
from modules.embedding.clip import TransformersClipEmbedder

__all__ = ["ClipEmbedder", "TransformersClipEmbedder"]
