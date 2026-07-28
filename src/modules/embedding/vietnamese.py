"""Vietnamese caption embedder: VLM caption -> sentence-transformers text vector.

Unlike SigLIP2/BEiT-3, this backend embeds *text*, not pixels — it exists to
cover what CLIP-style visual embedders miss (on-screen text, named entities,
long free-form Vietnamese queries), per the design in
``.claude/references/``. To stay a drop-in for the ``Embedder`` interface
(which is shape ``images -> vectors`` everywhere else, so ``indexing/embeddings.py``
needs no changes), ``embed_images`` internally captions each frame with a VLM
first, then embeds the caption text. Captions are cached to
``runs/<exp>/manifests/captions.jsonl`` (looked up by ``frame_id``) so re-runs
and retries don't re-call the (slow, network-bound) VLM for frames already
captioned.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import threading

from core.errors import EmbeddingError
from core.types import FrameRecord
from indexing.manifest import JsonlManifest
from modules.captioning.base import CaptioningModel
from modules.embedding.base import BatchCallback, BatchProgressLogger, Embedder

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
_DEFAULT_CAPTION_WORKERS = 8


class VietnameseEmbedder(Embedder):
    """Caption keyframes with a VLM, then embed the caption with a Vietnamese
    sentence-transformers model (BGE-M3 based, dot-product similarity)."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        captions_path: Path | None = None,
        captioner: CaptioningModel | None = None,
        caption_workers: int | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get("VIETNAMESE_EMBEDDING_MODEL", _DEFAULT_MODEL)
        self.device = device
        self.batch_size = batch_size
        self.captions_path = captions_path
        self.caption_workers = caption_workers or int(
            os.environ.get("CAPTION_WORKERS", _DEFAULT_CAPTION_WORKERS)
        )
        self._captioner = captioner
        self._model = None
        self._manifest: JsonlManifest | None = None
        self._manifest_lock = threading.Lock()
        self._caption_cache: dict[str, str] | None = None

    def embed_images(
        self, frames: list[FrameRecord], on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        """Caption each frame (cached, in parallel) then embed the captions as text.

        Captioning is a network round trip to vLLM (~2-3s each) — calling it
        sequentially for hundreds of thousands of keyframes would take days.
        vLLM's continuous batching handles many in-flight requests efficiently,
        so a thread pool (I/O-bound wait, not CPU-bound work) is enough to get
        real throughput without any change to vLLM itself. Submitting futures
        indexed by position (rather than ``executor.map``) lets us log
        progress as each caption completes, in whatever order threads finish,
        while still writing results back in the original frame order for the
        embed_texts() call below.
        """
        if not frames:
            return []
        self._load_caption_cache()
        progress = BatchProgressLogger(LOGGER, f"{self.model_name} (captioning)", len(frames))
        captions: list[str | None] = [None] * len(frames)
        with ThreadPoolExecutor(max_workers=self.caption_workers) as executor:
            future_to_index = {
                executor.submit(self._caption_for, frame): i for i, frame in enumerate(frames)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                captions[index] = future.result()
                progress.advance(1)
        return self._embed_texts(captions, frames=frames, on_batch=on_batch)

    def embed_text(self, query: str) -> list[float]:
        """Embed a single text query."""
        return self._embed_texts([query])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings (captions or queries)."""
        return self._embed_texts(texts)

    def _embed_texts(
        self,
        texts: list[str],
        frames: list[FrameRecord] | None = None,
        on_batch: BatchCallback | None = None,
    ) -> list[list[float]]:
        """Shared text-embedding loop; reports each batch via ``on_batch`` when
        ``frames`` (same length/order as ``texts``) is given, so embed_images()
        can checkpoint progress the same way SigLIP/BEiT-3 do."""
        model = self._load_model()
        progress = (
            BatchProgressLogger(LOGGER, f"{self.model_name} (encoding)", len(texts))
            if frames is not None
            else None
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings = model.encode(
                batch, batch_size=len(batch), normalize_embeddings=True, show_progress_bar=False
            )
            batch_vectors = embeddings.tolist()
            vectors.extend(batch_vectors)
            if on_batch is not None and frames is not None:
                on_batch(frames[start : start + self.batch_size], batch_vectors)
            if progress is not None:
                progress.advance(len(batch))
        return vectors

    def _caption_for(self, frame: FrameRecord) -> str:
        """Return the (cached or freshly generated) caption for one frame.

        Called concurrently from multiple threads (see embed_images); the
        cache read above is lock-free (dict reads are safe under the GIL),
        but writing the new caption to both the manifest file and the
        in-memory cache is serialized so concurrent appends never interleave
        mid-line in captions.jsonl.
        """
        cache = self._caption_cache
        if cache is not None and frame.frame_id in cache:
            return cache[frame.frame_id]

        captioner = self._load_captioner()
        caption = captioner.caption(frame.frame_path)

        with self._manifest_lock:
            manifest = self._load_manifest()
            if manifest is not None:
                manifest.append(
                    {
                        "frame_id": frame.frame_id,
                        "video_id": frame.video_id,
                        "caption": caption,
                    }
                )
            if cache is not None:
                cache[frame.frame_id] = caption
        return caption

    def _load_manifest(self) -> JsonlManifest | None:
        if self.captions_path is None:
            return None
        if self._manifest is None:
            self._manifest = JsonlManifest(self.captions_path)
        return self._manifest

    def _load_caption_cache(self) -> dict[str, str]:
        """Load ``captions.jsonl`` into memory once, keyed by ``frame_id``.

        Reading the whole manifest per-frame would be O(N^2) over hundreds of
        thousands of keyframes; loading it once up front keeps lookups O(1).
        """
        if self._caption_cache is not None:
            return self._caption_cache
        manifest = self._load_manifest()
        cache: dict[str, str] = {}
        if manifest is not None:
            for row in manifest.read_all():
                frame_id = row.get("frame_id")
                caption = row.get("caption")
                if frame_id is not None and caption is not None:
                    cache[str(frame_id)] = str(caption)
        self._caption_cache = cache
        return cache

    def _load_captioner(self) -> CaptioningModel:
        if self._captioner is not None:
            return self._captioner
        from modules.captioning.vllm import VllmCaptioningModel

        self._captioner = VllmCaptioningModel()
        return self._captioner

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "Install sentence-transformers before using the Vietnamese embedder."
            ) from exc

        resolved_device = _resolve_device(self.device)
        model = SentenceTransformer(self.model_name, device=resolved_device)
        model.max_seq_length = 2048
        self._model = model
        return model


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return requested
