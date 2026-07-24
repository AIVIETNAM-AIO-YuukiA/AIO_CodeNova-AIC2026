"""Embedding interface and shared helpers for all backends."""

from __future__ import annotations

from collections.abc import Callable
import logging
import time

from core.errors import EmbeddingError
from core.types import FrameRecord

# Called after each internal batch with (batch_frames, batch_vectors) so the
# caller can checkpoint progress to disk without waiting for the whole
# embed_images() call (which can cover hundreds of thousands of frames) to
# return. Optional — backends must accept it but callers may pass None.
BatchCallback = Callable[[list[FrameRecord], list[list[float]]], None]


class Embedder:
    """Interface for image/text embedding backends."""

    def embed_images(
        self, frames: list[FrameRecord], on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        """Embed image frames, optionally reporting each internal batch via ``on_batch``."""
        raise NotImplementedError

    def embed_text(self, query: str) -> list[float]:
        """Embed a text query."""
        raise NotImplementedError


class BatchProgressLogger:
    """Logs periodic progress through a large batch loop.

    ``embed_images()`` is called once with the full frame list (hundreds of
    thousands for a real dataset) and the caller (indexing/embeddings.py)
    only logs once the whole call returns — so without this, a run can sit
    silent for hours even while actively processing on GPU. Logs at most
    every ``min_interval_seconds`` to avoid flooding the log for small batch
    sizes.
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
    """Return the projected embedding tensor across Transformers versions.

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
        raise EmbeddingError("CUDA is not available; embedding requires a GPU in this project.")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise EmbeddingError(f"Requested device '{requested}' but CUDA is not available.")
    return torch.device(requested)
