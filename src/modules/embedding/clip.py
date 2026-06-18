"""Transformers-backed CLIP embedder."""

from __future__ import annotations

from pathlib import Path

from core.errors import EmbeddingError
from core.types import FrameRecord
from modules.embedding.base import ClipEmbedder


class TransformersClipEmbedder(ClipEmbedder):
    """CLIP image/text embeddings backed by Hugging Face Transformers."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32) -> None:
        self.model_name = normalize_clip_model_name(model_name)
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._torch = None

    def embed_images(self, frames: list[FrameRecord]) -> list[list[float]]:
        """Embed frames with CLIP image features."""
        if not frames:
            return []
        try:
            from PIL import Image
        except ImportError as exc:
            raise EmbeddingError("Install Pillow before embedding images.") from exc

        model, processor, torch, device = self._load()
        vectors: list[list[float]] = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            images = [Image.open(Path(frame.frame_path)).convert("RGB") for frame in batch]
            try:
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    features = clip_features_tensor(model.get_image_features(**inputs))
                    features = torch.nn.functional.normalize(features, p=2, dim=-1)
                vectors.extend(features.detach().cpu().numpy().astype("float32").tolist())
            finally:
                for image in images:
                    image.close()
        return vectors

    def embed_text(self, query: str) -> list[float]:
        """Embed a text query with CLIP text features."""
        model, processor, torch, device = self._load()
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = clip_features_tensor(model.get_text_features(**inputs))
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features.squeeze(0).detach().cpu().numpy().astype("float32").tolist()

    def _load(self):
        if self._model is not None:
            return self._model, self._processor, self._torch, self._device

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise EmbeddingError(
                "Install torch and transformers before running CLIP embeddings."
            ) from exc

        device = resolve_torch_device(torch, self.device)
        processor = CLIPProcessor.from_pretrained(self.model_name)
        model = CLIPModel.from_pretrained(self.model_name).eval().to(device)
        self._model = model
        self._processor = processor
        self._torch = torch
        self._device = device
        return model, processor, torch, device


def normalize_clip_model_name(model_name: str) -> str:
    """Map local short names to Hugging Face model IDs."""
    aliases = {
        "clip-vit-b-32": "openai/clip-vit-base-patch32",
        "vit-b-32": "openai/clip-vit-base-patch32",
        "clip-vit-l-14": "openai/clip-vit-large-patch14",
        "vit-l-14": "openai/clip-vit-large-patch14",
    }
    return aliases.get(model_name.lower(), model_name)


def clip_features_tensor(features):
    """Return the projected CLIP embedding tensor across Transformers versions.

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
        raise EmbeddingError(
            "CUDA is not available; CLIP embedding requires a GPU in this project."
        )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise EmbeddingError(f"Requested device '{requested}' but CUDA is not available.")
    return torch.device(requested)
