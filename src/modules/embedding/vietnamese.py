"""Embedder Vietnamese: VLM sinh caption -> sentence-transformers embed thanh vector.

Khac voi SigLIP2/BEiT-3, backend nay embed *text*, khong phai pixel - dung
de bu cho nhung thu CLIP-style bo sot (chu tren man hinh, ten rieng, cau
query tieng Viet dai). De van khop interface ``Embedder`` (dang
``images -> vectors`` giong cac backend khac, ``indexing/embeddings.py``
khong can sua gi), ``embed_images`` se tu goi VLM caption tung frame truoc,
roi embed noi dung caption do. Caption duoc cache vao
``runs/<exp>/manifests/captions.jsonl`` (tra theo ``frame_id``) de chay lai
khong phai goi VLM (cham, qua mang) cho nhung frame da co caption roi.
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
    """Caption keyframe bang VLM, roi embed caption bang model sentence-transformers
    tieng Viet (nen tang BGE-M3, do tuong dong bang dot-product)."""

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
        """Caption tung frame (co cache, chay song song) roi embed caption thanh vector.

        Goi caption la 1 luot goi mang toi vLLM (~2-3s moi lan) - goi tuan
        tu cho hang tram nghin keyframe se mat hang ngay. vLLM ho tro
        continuous batching xu ly nhieu request cung luc rat hieu qua, nen
        chi can thread pool (cho I/O, khong ton CPU) la du de dat toc do
        thuc te, khong can sua gi ben vLLM. Dung future danh so theo vi
        tri (thay vi ``executor.map``) de log tien do ngay khi tung
        caption xong bat ke thread nao hoan thanh truoc, nhung van ghi
        ket qua ve dung thu tu frame ban dau cho ham embed_texts() ben
        duoi.
        """
        if not frames:
            return []
        self._load_caption_cache()
        progress = BatchProgressLogger(LOGGER, f"{self.model_name} (captioning)", len(frames))
        captions: list[str | None] = [None] * len(frames)
        failed = 0
        with ThreadPoolExecutor(max_workers=self.caption_workers) as executor:
            future_to_index = {
                executor.submit(self._caption_for, frame): i for i, frame in enumerate(frames)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    captions[index] = future.result()
                except Exception:
                    # One frame the VLM refuses (a 400 from a content filter, a
                    # transient upstream fault) must not discard the whole
                    # batch's captions. Embed it as empty text and move on; a
                    # later run retries it, since it never reaches captions.jsonl.
                    LOGGER.exception("Captioning failed for %s", frames[index].frame_path)
                    captions[index] = ""
                    failed += 1
                progress.advance(1)
        if failed:
            LOGGER.warning("%s/%s frames could not be captioned this pass", failed, len(frames))
        return self._embed_texts(captions, frames=frames, on_batch=on_batch)

    def embed_text(self, query: str) -> list[float]:
        """Embed 1 cau query."""
        return self._embed_texts([query])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed nhieu chuoi text cung luc (caption hoac query)."""
        return self._embed_texts(texts)

    def _embed_texts(
        self,
        texts: list[str],
        frames: list[FrameRecord] | None = None,
        on_batch: BatchCallback | None = None,
    ) -> list[list[float]]:
        """Vong lap embed text dung chung; bao tien do qua ``on_batch`` khi co
        ``frames`` (cung do dai/thu tu voi ``texts``), de embed_images() checkpoint
        tien do giong cach SigLIP/BEiT-3 lam."""
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
        """Tra ve caption cua 1 frame (lay tu cache neu co, khong thi tao moi).

        Ham nay duoc goi dong thoi tu nhieu thread (xem embed_images);
        doc cache khong can lock (doc dict an toan duoi GIL), nhung ghi
        caption moi vao ca file manifest lan cache trong bo nho thi phai
        khoa lai de cac lan ghi dong thoi khong bi dan xen giua dong
        trong captions.jsonl.
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
        """Nap ``captions.jsonl`` vao bo nho 1 lan, tra theo ``frame_id``.

        Doc lai toan bo manifest cho tung frame se thanh O(N^2) voi hang
        tram nghin keyframe; nap 1 lan tu dau giu tra cuu la O(1).
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
    # Dat VIETNAMESE_EMBEDDING_DEVICE=cpu de text encoder nay khong dung GPU.
    # Voi GPU it VRAM, no se tranh cho voi cac image embedder va reranker;
    # chay tren CPU van du nhe, khong anh huong toi do tre.
    override = os.environ.get("VIETNAMESE_EMBEDDING_DEVICE")
    if override:
        return override
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return requested
