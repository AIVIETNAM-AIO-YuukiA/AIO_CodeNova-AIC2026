"""Transformers-backed SigLIP 2 embedder.

SigLIP 2 is a strong multilingual vision-language encoder. The API mirrors CLIP
(``get_image_features`` / ``get_text_features``), with two specifics: a unified
``AutoProcessor`` and text padding to a fixed length (SigLIP was trained with
``padding="max_length"``). The model is loaded via ``AutoModel`` so the correct
backend class is picked per checkpoint.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from core.errors import EmbeddingError
from core.types import FrameRecord
from modules.embedding.base import (
    BatchCallback,
    BatchProgressLogger,
    Embedder,
    projected_features,
    resolve_torch_device,
)

LOGGER = logging.getLogger(__name__)

# SigLIP 2's text tower has a 64-token context. Longer queries are embedded
# with a sliding window and mean-pooled (window 64, stride 48) — same
# technique as the AIC_2025 reference project's SigLIP_embedding.py.
_MAX_TEXT_TOKENS = 64
_WINDOW_STRIDE = 48


class SiglipEmbedder(Embedder):
    """SigLIP 2 image/text embeddings backed by Hugging Face Transformers."""

    def __init__(
        self, model_name: str | None = None, device: str = "auto", batch_size: int = 32
    ) -> None:
        resolved = model_name or os.environ.get("SIGLIP2_EMBEDDING_MODEL", "siglip2")
        self.model_name = normalize_siglip_model_name(resolved)
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._torch = None

    def embed_images(
        self, frames: list[FrameRecord], on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        """Embed frames with SigLIP image features."""
        if not frames:
            return []
        try:
            from PIL import Image
        except ImportError as exc:
            raise EmbeddingError("Install Pillow before embedding images.") from exc

        model, processor, torch, device = self._load()
        vectors: list[list[float]] = []
        progress = BatchProgressLogger(LOGGER, self.model_name, len(frames))
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            images = [Image.open(Path(frame.frame_path)).convert("RGB") for frame in batch]
            try:
                inputs = processor(images=images, return_tensors="pt").to(device)
                with torch.no_grad():
                    features = projected_features(model.get_image_features(**inputs))
                    features = torch.nn.functional.normalize(features, p=2, dim=-1)
                batch_vectors = features.detach().cpu().numpy().astype("float32").tolist()
                vectors.extend(batch_vectors)
            finally:
                for image in images:
                    image.close()
            progress.advance(len(batch))
            if on_batch is not None:
                on_batch(batch, batch_vectors)
        return vectors

    def embed_text(self, query: str) -> list[float]:
        """Embed a text query with SigLIP text features.

        The query is lowercased first — SigLIP 2 was trained on lowercased
        text, and mixed-case queries measurably hurt recall. Queries longer
        than the 64-token context are embedded per sliding window and
        mean-pooled instead of silently truncated.
        """
        model, processor, torch, device = self._load()
        text = query.lower().strip()
        windows = _text_windows(processor.tokenizer, text)
        inputs = processor(
            text=windows,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=_MAX_TEXT_TOKENS,
        ).to(device)
        with torch.no_grad():
            features = projected_features(model.get_text_features(**inputs))
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            pooled = torch.nn.functional.normalize(features.mean(dim=0), p=2, dim=0)
        return pooled.detach().cpu().numpy().astype("float32").tolist()

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


def _text_windows(tokenizer, text: str) -> list[str]:
    """Split ``text`` into <=64-token windows (stride 48); short text passes through."""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= _MAX_TEXT_TOKENS:
        return [text]
    windows = []
    start = 0
    while start < len(token_ids):
        chunk = token_ids[start : start + _MAX_TEXT_TOKENS]
        windows.append(tokenizer.decode(chunk, skip_special_tokens=True))
        if start + _MAX_TEXT_TOKENS >= len(token_ids):
            break
        start += _WINDOW_STRIDE
    return windows


def normalize_siglip_model_name(model_name: str) -> str:
    """Map local short names to Hugging Face model IDs.

    ``siglip2`` / ``siglip2-so400m`` resolve to the so400m-patch14-384
    checkpoint the AIC_2025 reference project uses (dim 1152, 384px) — not
    the NaFlex variant, whose variable-resolution input path behaves
    differently under AutoModel.
    """
    aliases = {
        "siglip2": "google/siglip2-so400m-patch14-384",
        "siglip2-base": "google/siglip2-base-patch16-256",
        "siglip2-large": "google/siglip2-large-patch16-256",
        "siglip2-so400m": "google/siglip2-so400m-patch14-384",
    }
    return aliases.get(model_name.lower(), model_name)
