"""Transformers-backed SigLIP 2 embedder.

SigLIP 2 is a strong multilingual vision-language encoder. The API mirrors CLIP
(``get_image_features`` / ``get_text_features``), with two specifics: a unified
``AutoProcessor`` and text padding to a fixed length (SigLIP was trained with
``padding="max_length"``). The model is loaded via ``AutoModel`` so the correct
backend class is picked per checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from core.errors import EmbeddingError
from core.types import FrameRecord
from modules.embedding.base import Embedder, projected_features, resolve_torch_device


class SiglipEmbedder(Embedder):
    """SigLIP 2 image/text embeddings backed by Hugging Face Transformers."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32) -> None:
        self.model_name = normalize_siglip_model_name(model_name)
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._torch = None

    def embed_images(self, frames: list[FrameRecord]) -> list[list[float]]:
        """Embed frames with SigLIP image features."""
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
                inputs = processor(images=images, return_tensors="pt").to(device)
                with torch.no_grad():
                    features = projected_features(model.get_image_features(**inputs))
                    features = torch.nn.functional.normalize(features, p=2, dim=-1)
                vectors.extend(features.detach().cpu().numpy().astype("float32").tolist())
            finally:
                for image in images:
                    image.close()
        return vectors

    def embed_text(self, query: str) -> list[float]:
        """Embed a text query with SigLIP text features."""
        model, processor, torch, device = self._load()
        inputs = processor(text=[query], return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            features = projected_features(model.get_text_features(**inputs))
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features.squeeze(0).detach().cpu().numpy().astype("float32").tolist()

    def _load(self):
        if self._model is not None:
            return self._model, self._processor, self._torch, self._device

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise EmbeddingError(
                "Install torch and transformers before running SigLIP embeddings."
            ) from exc

        device = resolve_torch_device(torch, self.device)
        processor = AutoProcessor.from_pretrained(self.model_name)
        # Use AutoModel so the right class is chosen by ``model_type``: SigLIP 2
        # fixed-resolution checkpoints (e.g. *-patch16-256) report model_type
        # "siglip", while only the NaFlex variant is genuine "siglip2".
        model = AutoModel.from_pretrained(self.model_name).eval().to(device)
        self._model = model
        self._processor = processor
        self._torch = torch
        self._device = device
        return model, processor, torch, device


def normalize_siglip_model_name(model_name: str) -> str:
    """Map local short names to Hugging Face model IDs."""
    aliases = {
        "siglip2": "google/siglip2-large-patch16-256",
        "siglip2-base": "google/siglip2-base-patch16-256",
        "siglip2-large": "google/siglip2-large-patch16-256",
        "siglip2-so400m": "google/siglip2-so400m-patch16-naflex",
    }
    return aliases.get(model_name.lower(), model_name)
