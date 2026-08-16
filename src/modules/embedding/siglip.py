"""Embedder SigLIP 2, chay qua Hugging Face Transformers.

SigLIP 2 la vision-language encoder da ngon ngu. API giong CLIP
(``get_image_features`` / ``get_text_features``), khac o 2 diem: dung chung
``AutoProcessor`` va text duoc pad ve do dai co dinh (SigLIP duoc train voi
``padding="max_length"``). Model duoc nap qua ``AutoModel`` de tu chon dung
class backend theo tung checkpoint.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import time

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

# Text tower cua SigLIP 2 chi nhan toi da 64 token. Query dai hon se duoc
# cat thanh nhieu doan (stride 48) roi mean-pool thay vi bi cat cut.
_MAX_TEXT_TOKENS = 64
_WINDOW_STRIDE = 48

# Giam kich thuoc anh dau vao truoc khi dua qua processor.
_MAX_SOURCE_SIDE = 512

_IMAGE_SIZE = 384
_EMBED_DIM = 1152

_DEFAULT_MODEL = "google/siglip2-so400m-patch14-384"

# So luong thread giai ma anh song song.
_DECODE_WORKERS = int(os.environ.get("SIGLIP2_DECODE_WORKERS", "8"))

# Dat SIGLIP2_USE_TENSORRT=0 de ep dung PyTorch thay vi engine TensorRT.
_USE_TENSORRT = os.environ.get("SIGLIP2_USE_TENSORRT", "1").lower() not in ("0", "false")

# So giay nghi giua cac batch. Voi GPU laptop de nong, nghi ngan giua batch
# giup GPU khong bi throttle nhiet, tong throughput co the tang len.
_BATCH_COOLDOWN = float(os.environ.get("SIGLIP2_BATCH_COOLDOWN", "0"))


class SiglipEmbedder(Embedder):
    """Embedding anh/text SigLIP 2, chay qua Hugging Face Transformers."""

    def __init__(
        self, model_name: str | None = None, device: str = "auto", batch_size: int = 32
    ) -> None:
        self.model_name = model_name or os.environ.get("SIGLIP2_EMBEDDING_MODEL", _DEFAULT_MODEL)
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
        """Embed frame bang SigLIP image features."""
        if not frames:
            return []
        try:
            import PIL.Image  # noqa: F401  (import de kiem tra Pillow co san)
        except ImportError as exc:
            raise EmbeddingError("Install Pillow before embedding images.") from exc

        use_trt = self._trt_encoder is not None
        if use_trt:
            processor, torch, device = self._load_preprocessing()
            model = None
        else:
            model, processor, torch, device = self._load()

        def _load_image(frame: FrameRecord):
            return _open_downscaled(Path(frame.frame_path))

        vectors: list[list[float]] = []
        progress = BatchProgressLogger(LOGGER, self.model_name, len(frames))
        with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as executor:
            if use_trt:
                # TensorRT engine duoc build san cho 1 kich thuoc batch co dinh
                # (opt_batch), nen van goi theo lo o day.
                for start in range(0, len(frames), self.batch_size):
                    batch = frames[start : start + self.batch_size]
                    images = list(executor.map(_load_image, batch))
                    try:
                        inputs = processor(images=images, return_tensors="pt").to(device)
                        raw = self._trt_encoder.infer(inputs["pixel_values"].to(torch.float32))
                        features = torch.nn.functional.normalize(raw, p=2, dim=-1, eps=1e-8)
                        batch_vectors = features.detach().cpu().numpy().astype("float32").tolist()
                        vectors.extend(batch_vectors)
                    finally:
                        for image in images:
                            image.close()
                    progress.advance(len(batch))
                    if on_batch is not None:
                        on_batch(batch, batch_vectors)
                    if _BATCH_COOLDOWN:
                        time.sleep(_BATCH_COOLDOWN)
            else:
                # Khong TensorRT: xu ly tung anh mot (giong AIC_2025's
                # get_image_embedding) thay vi batch GPU thuc su.
                for frame in frames:
                    image = _load_image(frame)
                    try:
                        inputs = processor(images=image, return_tensors="pt").to(device)
                        with torch.no_grad(), _autocast(torch, device):
                            features = projected_features(model.get_image_features(**inputs))
                            features = torch.nn.functional.normalize(features, p=2, dim=-1, eps=1e-8)
                        vector = features.detach().cpu().numpy().astype("float32").tolist()
                        vectors.extend(vector)
                    finally:
                        image.close()
                    progress.advance(1)
                    if on_batch is not None:
                        on_batch([frame], vector)
                    if _BATCH_COOLDOWN:
                        time.sleep(_BATCH_COOLDOWN)
        return vectors

    def _export_onnx(self, onnx_path: Path) -> None:
        """Trace vision tower cua SigLIP2 va xuat ra file ONNX."""
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
        """Embed 1 cau query bang SigLIP text features."""
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
        """Tra ve (processor, torch, device) ma khong nap model PyTorch."""
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
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        # Nap model roi chuyen thang sang device bang .to(). Checkpoint lon
        # co the bi transformers dat tam len "meta device"; goi .to(device)
        # sau do se bao loi "Cannot copy out of meta tensor" - da kiem tra
        # cach nay khong dinh loi do.
        model = AutoModel.from_pretrained(
            self.model_name, dtype=dtype, device_map="auto", low_cpu_mem_usage=True
        ).to(device).eval()
        # torch.compile giam overhead Python giua cac lan goi lien tiep -
        # chi co loi khi khong dung TensorRT (nhanh hon nhieu, khong can compile).
        if hasattr(torch, "compile") and device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                LOGGER.warning("[%s] torch.compile khong kha dung, dung eager mode", self.model_name)
        self._model = model
        self._processor = processor
        self._torch = torch
        self._device = device
        return model, processor, torch, device


def _open_downscaled(path):
    """Mo anh va giam kich thuoc ve gan ``_MAX_SOURCE_SIDE``.

    ``draft()`` cho phep JPEG decoder xuat anh ty le 1/2, 1/4 hoac 1/8 ngay
    luc giai ma, re hon nhieu so voi giai ma full 1280x720 roi moi chay
    LANCZOS resize - khi dung TensorRT tren GPU, buoc resize nay chinh la
    nut that co chai (GPU chi dung ~10% memory vi phai cho). Phan resize
    con lai van dung LANCZOS nen chat luong dau ra khong doi voi input
    384px cua model.
    """
    from PIL import Image

    image = Image.open(path)
    if image.format == "JPEG":
        image.draft("RGB", (_MAX_SOURCE_SIDE, _MAX_SOURCE_SIDE))
    return _downscale(image.convert("RGB"))


def _downscale(image):
    """Thu nho anh sao cho canh dai nhat toi da 512px (dung LANCZOS)."""
    from PIL import Image

    if max(image.size) <= _MAX_SOURCE_SIDE:
        return image
    ratio = _MAX_SOURCE_SIDE / max(image.size)
    new_size = tuple(int(dim * ratio) for dim in image.size)
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    image.close()
    return resized


def _autocast(torch, device):
    """Bat autocast fp16 khi chay tren CUDA, khong lam gi o device khac."""
    if getattr(device, "type", str(device)) == "cuda" or str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    import contextlib

    return contextlib.nullcontext()


def _text_windows(tokenizer, text: str) -> list[str]:
    """Chia ``text`` thanh cac doan toi da 64 token (stride 48); text ngan giu nguyen."""
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
