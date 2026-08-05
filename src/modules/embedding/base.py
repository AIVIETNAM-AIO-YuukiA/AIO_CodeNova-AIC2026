"""Interface embedding va cac ham dung chung cho moi backend."""

from __future__ import annotations

from collections.abc import Callable
import logging
import time

from core.errors import EmbeddingError
from core.types import FrameRecord

# Duoc goi sau moi batch noi bo voi (batch_frames, batch_vectors) de noi
# goi co the checkpoint tien do ra dia ma khong phai cho ca ham
# embed_images() (co the xu ly hang tram nghin frame) tra ve. La optional -
# backend phai chap nhan tham so nay nhung noi goi co the truyen None.
BatchCallback = Callable[[list[FrameRecord], list[list[float]]], None]


class Embedder:
    """Interface cho cac backend embedding anh/text."""

    def embed_images(
        self, frames: list[FrameRecord], on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        """Embed cac frame anh, co the bao tien do tung batch qua ``on_batch``."""
        raise NotImplementedError

    def embed_text(self, query: str) -> list[float]:
        """Embed 1 cau query text."""
        raise NotImplementedError


class BatchProgressLogger:
    """Log tien do dinh ky trong 1 vong lap batch lon.

    ``embed_images()`` duoc goi 1 lan voi toan bo danh sach frame (hang
    tram nghin frame voi du lieu that), va noi goi (indexing/embeddings.py)
    chi log khi ca lan goi tra ve xong - nen neu khong co class nay, 1 lan
    chay co the im lang hang gio du dang xu ly that tren GPU. Chi log toi
    da moi ``min_interval_seconds`` de tranh log qua nhieu khi batch nho.
    """

    def __init__(
        self, logger: logging.Logger, label: str, total: int, min_interval_seconds: float = 10.0
    ) -> None:
        self._logger = logger
        self._label = label
        self._total = total
        self._min_interval = min_interval_seconds
        self._done = 0
        self._last_logged = 0.0

    def advance(self, count: int) -> None:
        """Record that ``count`` more items completed, logging if due."""
        self._done += count
        now = time.monotonic()
        is_last = self._done >= self._total
        if is_last or (now - self._last_logged) >= self._min_interval:
            pct = 100.0 * self._done / self._total if self._total else 100.0
            self._logger.info(
                "[%s] embedded %s/%s (%.1f%%)", self._label, self._done, self._total, pct
            )
            self._last_logged = now


def projected_features(features):
    """Tra ve tensor embedding da projected, dung duoc voi moi ban Transformers.

    Transformers 4.x tra thang tensor tu ``get_image_features`` va
    ``get_text_features``. Transformers 5.x tra ve
    ``BaseModelOutputWithPooling``, embedding nam trong ``pooler_output``.
    """
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def resolve_torch_device(torch, requested: str):
    """Xac dinh torch device; bat buoc phai co CUDA voi che do auto/cuda."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise EmbeddingError("CUDA is not available; embedding requires a GPU in this project.")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise EmbeddingError(f"Requested device '{requested}' but CUDA is not available.")
    return torch.device(requested)
