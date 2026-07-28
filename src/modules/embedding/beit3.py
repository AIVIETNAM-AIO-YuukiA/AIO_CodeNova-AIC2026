"""BEiT-3 large embedder (image-text retrieval, COCO fine-tune).

BEiT-3 has no HuggingFace ``transformers`` port, so this backend loads the
original Microsoft implementation vendored under ``_beit3_vendor/`` (source:
github.com/microsoft/unilm/tree/master/beit3, MIT license) together with a
vendored copy of ``torchscale`` (its only non-stdlib dependency besides
``timm``/``fairscale``, copied because upstream ``torchscale`` pins
``timm==0.4.12`` which conflicts with the rest of this project).

The vendor source uses flat, non-relative imports (``import utils``,
``import torchscale.xxx``) exactly as in the upstream repo, so it is loaded by
adding its directory to ``sys.path`` rather than as a regular Python package.

Uses ``beit3_large_patch16_384_retrieval`` with the COCO-retrieval checkpoint
(dim 1024, 384px, ImageNet-default normalization, bilinear resize).

Checkpoint and tokenizer (``beit3_large_patch16_384_coco_retrieval.pth``,
``beit3.spm``, not committed — see ``external/BEiT3/checkpoints/``) are
downloaded automatically on first use from
https://github.com/addf400/files/releases/download/beit3/.

Image embedding runs through a TensorRT engine (auto-exported/built on first
use, see ``tensorrt_runtime.py``) unless ``BEIT3_USE_TENSORRT=0``. Text
embedding always uses PyTorch.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import sys

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

# Parallel image decode workers.
_DECODE_WORKERS = int(os.environ.get("BEIT3_DECODE_WORKERS", "8"))

# Set BEIT3_USE_TENSORRT=0 to force PyTorch instead of the TensorRT engine.
_USE_TENSORRT = os.environ.get("BEIT3_USE_TENSORRT", "1").lower() not in ("0", "false")

_VENDOR_DIR = Path(__file__).resolve().parent / "_beit3_vendor"
_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "external" / "BEiT3" / "checkpoints"
_DEFAULT_CHECKPOINT_FILE = "beit3_large_patch16_384_coco_retrieval.pth"
_DEFAULT_SPM_FILE = "beit3.spm"
_RELEASE_BASE_URL = "https://github.com/addf400/files/releases/download/beit3"


def _checkpoint_dir() -> Path:
    override = os.environ.get("BEIT3_CHECKPOINT_DIR")
    return Path(override) if override else _DEFAULT_CHECKPOINT_DIR


def _checkpoint_file() -> str:
    return os.environ.get("BEIT3_CHECKPOINT_FILE", _DEFAULT_CHECKPOINT_FILE)


def _spm_file() -> str:
    return os.environ.get("BEIT3_SPM_FILE", _DEFAULT_SPM_FILE)


def _ensure_downloaded(path: Path) -> None:
    """Download ``path`` from the BEiT-3 release assets if it doesn't exist yet.

    Streams to a temp file first and renames on success, so a killed/failed
    download never leaves a corrupt file that looks present on retry.
    """
    if path.exists():
        return
    try:
        import httpx
    except ImportError as exc:
        raise EmbeddingError("Install httpx before running BEiT-3 embeddings.") from exc

    url = f"{_RELEASE_BASE_URL}/{path.name}"
    LOGGER.info("Downloading BEiT-3 asset %s from %s...", path.name, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
        tmp_path.rename(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise EmbeddingError(f"Failed to download {url}: {exc}") from exc
    LOGGER.info("Downloaded %s (%.1f MB)", path.name, path.stat().st_size / 1024 / 1024)


# XLM-R sentencepiece convention: piece ids are offset by +1, wrapped with
# <s>=0 ... </s>=2. Matches microsoft/unilm's XLMRobertaTokenizer usage.
_BOS_ID, _PAD_ID, _EOS_ID = 0, 1, 2
_MAX_TEXT_LEN = 64
_IMAGE_SIZE = 384
_EMBED_DIM = 1024


class Beit3Embedder(Embedder):
    """BEiT3-large-384 COCO-retrieval image/text embeddings (dim 1024)."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._sp = None
        self._transform = None
        self._torch = None
        self._torch_device = None
        self._trt_encoder = (
            TensorRTVisionEncoder(
                model_key="beit3-large",
                export_onnx_fn=self._export_onnx,
                input_name="image",
                output_name="vision_cls",
                output_dim=_EMBED_DIM,
                opt_batch=batch_size,
            )
            if _USE_TENSORRT
            else None
        )

    def embed_images(
        self, frames: list[FrameRecord], on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        """Embed frames with BEiT-3 vision features."""
        if not frames:
            return []
        try:
            from PIL import Image
        except ImportError as exc:
            raise EmbeddingError("Install Pillow before embedding images.") from exc

        use_trt = self._trt_encoder is not None
        if use_trt:
            transform, torch, device = self._load_preprocessing()
            model = None
        else:
            model, _, transform, torch, device = self._load()

        vectors: list[list[float]] = []
        progress = BatchProgressLogger(LOGGER, self.model_name, len(frames))

        def _load_tensor(frame: FrameRecord):
            with Image.open(Path(frame.frame_path)) as image:
                return transform(image.convert("RGB"))

        with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as executor:
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start : start + self.batch_size]
                tensors = list(executor.map(_load_tensor, batch))
                pixel_values = torch.stack(tensors).to(device)
                if use_trt:
                    vision_cls = self._trt_encoder.infer(pixel_values)
                else:
                    with torch.no_grad(), _autocast(torch, device):
                        vision_cls, _ = model(image=pixel_values, only_infer=True)
                batch_vectors = vision_cls.detach().cpu().numpy().astype("float32").tolist()
                vectors.extend(batch_vectors)
                progress.advance(len(batch))
                if on_batch is not None:
                    on_batch(batch, batch_vectors)
        return vectors

    def _export_onnx(self, onnx_path: Path) -> None:
        """Trace the BEiT-3 vision-only forward path and export it to ONNX."""
        model, _, _, torch, device = self._load()

        class _VisionOnly(torch.nn.Module):
            def __init__(self, wrapped) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, image):
                vision_cls, _ = self.wrapped(image=image, only_infer=True)
                return vision_cls

        wrapper = _VisionOnly(model).eval()
        dummy = torch.randn(1, 3, _IMAGE_SIZE, _IMAGE_SIZE, device=device)
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["image"],
            output_names=["vision_cls"],
            dynamic_axes={"image": {0: "batch"}, "vision_cls": {0: "batch"}},
            opset_version=18,
            do_constant_folding=True,
        )

    def embed_text(self, query: str) -> list[float]:
        """Embed a text query with BEiT-3 language features."""
        model, sp, _, torch, device = self._load()
        input_ids, padding_mask = _tokenize(sp, query, torch)
        input_ids = input_ids.to(device)
        padding_mask = padding_mask.to(device)
        with torch.no_grad(), _autocast(torch, device):
            _, language_cls = model(
                text_description=input_ids, padding_mask=padding_mask, only_infer=True
            )
        return language_cls.squeeze(0).detach().cpu().numpy().astype("float32").tolist()

    def _load_preprocessing(self):
        """Return (transform, torch, device) without loading the PyTorch model."""
        if self._transform is not None:
            return self._transform, self._torch, self._torch_device

        try:
            import torch
            from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            from torchvision import transforms
        except ImportError as exc:
            raise EmbeddingError(
                "Install torch, torchvision, and timm before running BEiT-3 embeddings."
            ) from exc

        device = resolve_torch_device(torch, self.device)
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (_IMAGE_SIZE, _IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        self._transform = transform
        self._torch = torch
        self._torch_device = device
        return transform, torch, device

    def _load(self):
        if self._model is not None:
            return self._model, self._sp, self._transform, self._torch, self._torch_device

        checkpoint_dir = _checkpoint_dir()
        checkpoint_path = checkpoint_dir / _checkpoint_file()
        spm_path = checkpoint_dir / _spm_file()
        _ensure_downloaded(checkpoint_path)
        _ensure_downloaded(spm_path)

        try:
            import sentencepiece as spm
            import torch
            from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            from torchvision import transforms
        except ImportError as exc:
            raise EmbeddingError(
                "Install torch, torchvision, timm, and sentencepiece before running "
                "BEiT-3 embeddings."
            ) from exc

        if str(_VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(_VENDOR_DIR))
        try:
            import modeling_finetune
        except ImportError as exc:
            raise EmbeddingError(
                "Install torchscale and fairscale before running BEiT-3 embeddings "
                f"(vendored BEiT-3 source expected at {_VENDOR_DIR})."
            ) from exc

        device = resolve_torch_device(torch, self.device)
        model = modeling_finetune.beit3_large_patch16_384_retrieval()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.eval().to(device)

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (_IMAGE_SIZE, _IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        sp = spm.SentencePieceProcessor(model_file=str(spm_path))

        self._model = model
        self._sp = sp
        self._transform = transform
        self._torch = torch
        self._torch_device = device
        return model, sp, transform, torch, device


def _autocast(torch, device):
    """fp16 autocast on CUDA (PyTorch fallback path only), no-op elsewhere."""
    if getattr(device, "type", str(device)) == "cuda" or str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    import contextlib

    return contextlib.nullcontext()


def _tokenize(sp, text: str, torch):
    """XLM-R style tokenization: <s> piece_ids+1 ... </s> <pad>*, plus a padding mask."""
    pieces = sp.encode(text, out_type=int)
    ids = [_BOS_ID, *(piece + 1 for piece in pieces), _EOS_ID][:_MAX_TEXT_LEN]
    pad_len = _MAX_TEXT_LEN - len(ids)
    padding_mask = [0] * len(ids) + [1] * pad_len
    ids = ids + [_PAD_ID] * pad_len
    return torch.tensor([ids]), torch.tensor([padding_mask])
