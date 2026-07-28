"""Transformers-backed SigLIP 2 embedder.

SigLIP 2 is a strong multilingual vision-language encoder. The API mirrors CLIP
(``get_image_features`` / ``get_text_features``), with two specifics: a unified
``AutoProcessor`` and text padding to a fixed length (SigLIP was trained with
``padding="max_length"``). The model is loaded via ``AutoModel`` so the correct
backend class is picked per checkpoint.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
from modules.embedding.tensorrt_runtime import TensorRTVisionEncoder

LOGGER = logging.getLogger(__name__)

# SigLIP 2's text tower has a 64-token context. Longer queries are split into
# windows (stride 48) and mean-pooled instead of being truncated.
_MAX_TEXT_TOKENS = 64
_WINDOW_STRIDE = 48

# Downscale oversized source images before the processor.
_MAX_SOURCE_SIDE = 512

_IMAGE_SIZE = 384
_EMBED_DIM = 1152

# Parallel image decode workers.
_DECODE_WORKERS = int(os.environ.get("SIGLIP2_DECODE_WORKERS", "8"))

# Set SIGLIP2_USE_TENSORRT=0 to force PyTorch instead of the TensorRT engine.
_USE_TENSORRT = os.environ.get("SIGLIP2_USE_TENSORRT", "1").lower() not in ("0", "false")


class SiglipEmbedder(Embedder):
    """SigLIP 2 image/text embeddings backed by Hugging Face Transformers."""

    def __init__(
        self, model_name: str | None = None, device: str = "auto", batch_size: int = 32
    ) -> None:
        resolved = model_name or os.environ.get("SIGLIP2_EMBEDDING_MODEL", "siglip2-so400m")
        self.model_name = normalize_siglip_model_name(resolved)
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._torch = None
        self._trt_encoder = (
            TensorRTVisionEncoder(
                model_key=self.model_name.replace("/", "__"),
                export_onnx_fn=self._export_onnx,
                input_name="pixel_values",
                output_name="select_2",
                output_dim=_EMBED_DIM,
                opt_batch=batch_size,
            )
            if _USE_TENSORRT
            else None
        )

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

        use_trt = self._trt_encoder is not None
        if use_trt:
            processor, torch, device = self._load_preprocessing()
            model = None
        else:
            model, processor, torch, device = self._load()

        def _load_image(frame: FrameRecord):
            return _downscale(Image.open(Path(frame.frame_path)).convert("RGB"))

        vectors: list[list[float]] = []
        progress = BatchProgressLogger(LOGGER, self.model_name, len(frames))
        with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as executor:
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start : start + self.batch_size]
                images = list(executor.map(_load_image, batch))
                try:
                    inputs = processor(images=images, return_tensors="pt").to(device)
                    if use_trt:
                        raw = self._trt_encoder.infer(inputs["pixel_values"].to(torch.float32))
                        features = torch.nn.functional.normalize(raw, p=2, dim=-1)
                    else:
                        with torch.no_grad(), _autocast(torch, device):
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

    def _export_onnx(self, onnx_path: Path) -> None:
        """Trace the SigLIP2 vision tower and export it to ONNX."""
        model, _, torch, device = self._load()
        dummy = torch.randn(1, 3, _IMAGE_SIZE, _IMAGE_SIZE, device=device)
        torch.onnx.export(
            model.vision_model,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["pooler_output"],
            dynamic_axes={"pixel_values": {0: "batch"}, "pooler_output": {0: "batch"}},
            opset_version=18,
            do_constant_folding=True,
        )

    def embed_text(self, query: str) -> list[float]:
        """Embed a text query with SigLIP text features."""
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
        with torch.no_grad(), _autocast(torch, device):
            features = projected_features(model.get_text_features(**inputs))
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            pooled = torch.nn.functional.normalize(features.mean(dim=0), p=2, dim=0)
        return pooled.detach().cpu().numpy().astype("float32").tolist()

    def _load_preprocessing(self):
        """Return (processor, torch, device) without loading the PyTorch model."""
        if self._processor is not None:
            return self._processor, self._torch, self._device

        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise EmbeddingError(
                "Install torch and transformers before running SigLIP embeddings."
            ) from exc

        device = resolve_torch_device(torch, self.device)
        processor = AutoProcessor.from_pretrained(self.model_name)
        self._processor = processor
        self._torch = torch
        self._device = device
        return processor, torch, device

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
        model = AutoModel.from_pretrained(self.model_name).eval().to(device)
        self._model = model
        self._processor = processor
        self._torch = torch
        self._device = device
        return model, processor, torch, device


def _downscale(image):
    """Shrink an image so its longest side is at most 512px (LANCZOS)."""
    from PIL import Image

    if max(image.size) <= _MAX_SOURCE_SIDE:
        return image
    ratio = _MAX_SOURCE_SIDE / max(image.size)
    new_size = tuple(int(dim * ratio) for dim in image.size)
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    image.close()
    return resized


def _autocast(torch, device):
    """fp16 autocast on CUDA, no-op elsewhere."""
    if getattr(device, "type", str(device)) == "cuda" or str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    import contextlib

    return contextlib.nullcontext()


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

    ``siglip2`` / ``siglip2-so400m`` resolve to the fixed-resolution
    so400m-patch14-384 checkpoint (dim 1152), not the NaFlex variant, whose
    variable-resolution input path behaves differently under AutoModel.
    """
    aliases = {
        "siglip2": "google/siglip2-so400m-patch14-384",
        "siglip2-base": "google/siglip2-base-patch16-256",
        "siglip2-large": "google/siglip2-large-patch16-256",
        "siglip2-so400m": "google/siglip2-so400m-patch14-384",
    }
    return aliases.get(model_name.lower(), model_name)
