"""Image/text embedding models."""

from codenova.modules.embedding.base import ClipEmbedder
from codenova.modules.embedding.clip import TransformersClipEmbedder

__all__ = ["ClipEmbedder", "TransformersClipEmbedder"]
