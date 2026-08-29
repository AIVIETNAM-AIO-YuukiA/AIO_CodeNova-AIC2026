"""Embedder Jina CLIP v2 (jinaai/jina-clip-v2, dim 1024)."""

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
    resolve_torch_device,
)
from modules.embedding.tensorrt_runtime import TensorRTVisionEncoder

LOGGER = logging.getLogger(__name__)

_MODEL_ID = "jinaai/jina-clip-v2"
_EMBED_DIM = 1024
_QUERY_TASK = "retrieval.query"
_IMAGE_SIZE = 512

_DECODE_WORKERS = int(os.environ.get("JINA_DECODE_WORKERS", "8"))

# Dat JINA_USE_TENSORRT=0 de ep dung PyTorch thay vi engine TensorRT.
_USE_TENSORRT = os.environ.get("JINA_USE_TENSORRT", "1").lower() not in ("0", "false")


class JinaClipEmbedder(Embedder):
    """Embedding anh/text Jina CLIP v2, load qua transformers AutoModel."""

    def __init__(
        self, model_name: str | None = None, device: str = "auto", batch_size: int = 32
    ) -> None:
        resolved = model_name or os.environ.get("JINA_EMBEDDING_MODEL", _MODEL_ID)
        self.model_name = _MODEL_ID if "/" not in resolved else resolved
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._trt_encoder = (
            TensorRTVisionEncoder(
                model_key=self.model_name.replace("/", "__"),
                export_onnx_fn=self._export_onnx,
                input_name="pixel_values",
                output_name="image_embeds",
                output_dim=_EMBED_DIM,
                opt_batch=batch_size,
            )
            if _USE_TENSORRT
            else None
        )

    def embed_images(
        self, frames: list[FrameRecord], on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        if not frames:
            return []
        try:
            from PIL import Image
        except ImportError as exc:
            raise EmbeddingError("Install Pillow before embedding images.") from exc

        use_trt = self._trt_encoder is not None
        if use_trt:
            processor, torch, device = self._load_preprocessing()
        else:
            _, processor, torch, device = self._load()

        def _load_image(frame: FrameRecord):
            return Image.open(Path(frame.frame_path)).convert("RGB")

        vectors: list[list[float]] = []
        progress = BatchProgressLogger(LOGGER, self.model_name, len(frames))
        with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as executor:
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start : start + self.batch_size]
                images = list(executor.map(_load_image, batch))
                try:
                    if use_trt:
                        pixel_values = processor(images=images, return_tensors="pt")[
                            "pixel_values"
                        ].to(device, dtype=torch.float32)
                        raw = self._trt_encoder.infer(pixel_values)
                    else:
                        model, _, _, _ = self._load()
                        with torch.inference_mode():
                            raw = model.encode_image(images, batch_size=len(images))
                    batch_vectors = _l2_normalize(raw)
                    batch_vectors = self._repair_non_finite(batch_vectors, images, torch)
                    vectors.extend(batch_vectors)
                finally:
                    for image in images:
                        image.close()
                progress.advance(len(batch))
                if on_batch is not None:
                    on_batch(batch, batch_vectors)
        return vectors

    def _export_onnx(self, onnx_path: Path) -> None:
        """Trace ham get_image_features cua Jina va xuat ra file ONNX.

        Model that (encode_image()) khong trace duoc truc tiep vi nhan
        list PIL.Image lam input, khong phai tensor. get_image_features()
        la ham noi bo dung chung vision tower, nhan thang pixel_values
        (batch, 3, 512, 512) da preprocess - cung 1 output voi encode_image,
        chi khac o buoc preprocessing nam ngoai, nen trace duoc binh thuong.
        """
        model, _, torch, device = self._load()

        class _VisionOnly(torch.nn.Module):
            def __init__(self, wrapped) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, pixel_values):
                return self.wrapped.get_image_features(pixel_values=pixel_values)

        wrapper = _VisionOnly(model).eval()
        dummy = torch.randn(1, 3, _IMAGE_SIZE, _IMAGE_SIZE, device=device)
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
            opset_version=18,
            do_constant_folding=True,
        )

    def _repair_non_finite(self, batch_vectors, images, torch):
        """Embed lai bang fp32 neu vector bi NaN/Inf.

        Vision tower EVA thinh thoang tran so o fp16 (khoang 1/5000 frame
        trong bo du lieu nay), ra vector toan NaN se lam hong index. Embed
        lai dung nhung anh loi bang fp32 rat re, dam bao moi frame tim
        kiem duoc.
        """
        import numpy as np

        array = np.asarray(batch_vectors, dtype=np.float32)
        bad = ~np.isfinite(array).all(axis=-1)
        if not bad.any():
            return batch_vectors

        model, _, torch_ref, device = self._load()
        indices = np.flatnonzero(bad).tolist()
        LOGGER.warning("Re-embedding %s non-finite vector(s) in fp32", len(indices))
        try:
            model.float()
            with torch_ref.inference_mode():
                raw = model.encode_image([images[i] for i in indices], batch_size=len(indices))
            repaired = _l2_normalize(raw)
        finally:
            model.to(
                device, dtype=torch_ref.float16 if device.type == "cuda" else torch_ref.float32
            )

        for slot, vector in zip(indices, repaired):
            batch_vectors[slot] = vector
        return batch_vectors

    def embed_text(self, query: str) -> list[float]:
        # Text encoding luon chay PyTorch: moi lan chi embed 1 query, khong
        # co loi ich gi tu TensorRT batching.
        model, _, torch, _ = self._load()
        with torch.inference_mode():
            raw = model.encode_text([query], task=_QUERY_TASK)
        return _l2_normalize(raw)[0]

    def _load_preprocessing(self):
        """Tra ve (processor, torch, device) ma khong nap model PyTorch."""
        if self._processor is not None:
            return self._processor, self._torch, self._device

        try:
            import torch
            from transformers import AutoImageProcessor
        except ImportError as exc:
            raise EmbeddingError(
                "Install torch and transformers before running Jina CLIP embeddings."
            ) from exc

        device = resolve_torch_device(torch, self.device)
        processor = AutoImageProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._processor = processor
        self._torch = torch
        self._device = device
        return processor, torch, device

    def _load(self):
        if self._model is not None:
            return self._model, self._processor, self._torch, self._device

        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise EmbeddingError(
                "Install torch and transformers before running Jina CLIP embeddings."
            ) from exc

        device = resolve_torch_device(torch, self.device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        # Nap model roi chuyen thang sang device bang .to(). Checkpoint lon
        # co the bi transformers dat tam len "meta device"; goi .to(device)
        # sau do se bao loi "Cannot copy out of meta tensor" - da kiem tra
        # cach nay khong dinh loi do.
        model = (
            AutoModel.from_pretrained(self.model_name, trust_remote_code=True, dtype=dtype)
            .to(device)
            .eval()
        )

        # processor co the da duoc _load_preprocessing() nap truoc do (khi
        # dung duong TensorRT); chi nap moi neu chua co.
        if self._processor is None:
            self._processor, _, _ = self._load_preprocessing()

        self._model = model
        self._torch = torch
        self._device = device
        return model, self._processor, torch, device


def _l2_normalize(raw) -> list[list[float]]:
    """Chuan hoa L2 dau ra encoder (numpy array hoac tensor) thanh list float32.

    TensorRT tra ve tensor nam tren CUDA, numpy khong doc truc tiep duoc
    nen phai chuyen ve CPU truoc.
    """
    import numpy as np

    if hasattr(raw, "detach"):
        raw = raw.detach().cpu()
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return (array / (norms + 1e-12)).tolist()
