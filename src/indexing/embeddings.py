"""Embedding pipeline stage (incremental, multi-model)."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
import time

from core.errors import EmbeddingError
from core.logging import get_logger

from config.settings import Experiment
from core.paths import require_experiment_frame_path
from core.types import FrameRecord
from modules.embedding import build_embedder, resolve_embedding_models
from indexing.embedding_paths import (
    checkpoint_frame_ids_path,
    checkpoint_vectors_path,
    frame_ids_path,
    vectors_path,
)
from indexing.manifest import JsonlManifest, ManifestError
from indexing.state import JobState

LOGGER = get_logger(__name__)

# How often (seconds) an in-progress model flushes its checkpoint to disk.
# embed_images() can run for hours over hundreds of thousands of frames;
# checkpointing lets a killed/interrupted run resume mid-model instead of
# redoing all of it, at the cost of a small periodic disk write.
_CHECKPOINT_INTERVAL_SECONDS = 30.0

# Frames handed to embed_images() per call. Caps the embedder's peak memory
# (see the chunked loop in embed_frames) without changing what gets embedded.
_EMBED_CHUNK_SIZE = int(os.environ.get("EMBED_CHUNK_SIZE", "2000"))


class _CheckpointWriter:
    """Buffers embedded (frame_id, vector) pairs and periodically flushes them
    to a ``*.checkpoint.npz`` / ``*.checkpoint.json`` pair, so a run killed
    mid-model can resume from the last flush instead of from scratch.

    Vectors are kept as a NumPy array, not a Python list of lists: with
    hundreds of thousands of 1152-dim float vectors, each Python float object
    costs ~28 bytes vs. 4 bytes packed in a float32 array — a list-of-lists
    of e.g. 175K x 1152 floats runs to ~7GB of pure object overhead, most of
    a laptop's RAM, for data a NumPy array holds in under 1GB.
    """

    def __init__(self, np, checkpoint_vectors_path, checkpoint_ids_path, prior_ids, prior_vectors):
        self._np = np
        self._vectors_path = checkpoint_vectors_path
        self._ids_path = checkpoint_ids_path
        self._ids: list[str] = list(prior_ids)
        self._vectors = np.asarray(prior_vectors, dtype="float32") if len(prior_vectors) else None
        self._pending: list = []
        self._last_flush = time.monotonic()

    def add_batch(self, frames: list[FrameRecord], vectors: list[list[float]]) -> None:
        self._ids.extend(frame.frame_id for frame in frames)
        self._pending.extend(vectors)
        now = time.monotonic()
        if now - self._last_flush >= _CHECKPOINT_INTERVAL_SECONDS:
            self.flush()
            self._last_flush = now

    def _merge_pending(self) -> None:
        """Fold buffered raw batches into the NumPy array, then drop the list."""
        if not self._pending:
            return
        new_array = self._np.asarray(self._pending, dtype="float32")
        self._vectors = (
            new_array
            if self._vectors is None
            else self._np.concatenate([self._vectors, new_array], axis=0)
        )
        self._pending = []

    def flush(self) -> None:
        """Write vectors/ids atomically (temp file + rename)."""
        if not self._ids:
            return
        self._merge_pending()
        vectors_tmp = self._vectors_path.with_name(self._vectors_path.name + ".tmp.npz")
        ids_tmp = self._ids_path.with_suffix(self._ids_path.suffix + ".tmp")
        self._np.savez(vectors_tmp, embeddings=self._vectors)
        ids_tmp.write_text(json.dumps(self._ids) + "\n", encoding="utf-8")
        vectors_tmp.replace(self._vectors_path)
        ids_tmp.replace(self._ids_path)


def _load_checkpoint(np, checkpoint_vectors_path, checkpoint_ids_path):
    """Return (ids, vectors) from a prior interrupted run's checkpoint, or empty lists.

    ``vectors`` stays a NumPy array (not ``.tolist()``) — callers that only
    need it to seed ``_CheckpointWriter`` or build the final ``.npz`` should
    never round-trip through a Python list of floats.
    """
    if not (checkpoint_vectors_path.exists() and checkpoint_ids_path.exists()):
        return [], np.empty((0, 0), dtype="float32")
    ids = json.loads(checkpoint_ids_path.read_text(encoding="utf-8"))
    vectors = np.load(checkpoint_vectors_path)["embeddings"].astype("float32")
    return ids, vectors


def _discard_checkpoint(checkpoint_vectors_path, checkpoint_ids_path) -> None:
    checkpoint_vectors_path.unlink(missing_ok=True)
    checkpoint_ids_path.unlink(missing_ok=True)


def _validate_artifact(np, vectors, frame_ids: list[str], known_ids: set[str]) -> None:
    if vectors.ndim != 2:
        raise RuntimeError(f"Embedding array must be 2-D, got shape={vectors.shape}")
    if len(vectors) != len(frame_ids):
        raise RuntimeError(
            f"Embedding vector/ID mismatch vectors={len(vectors)} ids={len(frame_ids)}"
        )
    if len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError("Embedding artifact contains duplicate frame IDs")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding artifact contains NaN or Inf")
    unknown = set(frame_ids) - known_ids
    if unknown:
        raise RuntimeError(f"Embedding artifact references {len(unknown)} unknown frames")


def _publish_artifact(np, vector_path, ids_path, vectors, frame_ids: list[str]) -> None:
    """Validate temporary files before replacing the legacy-compatible pair."""
    vector_tmp = vector_path.with_name(vector_path.name + ".publish.tmp.npz")
    ids_tmp = ids_path.with_name(ids_path.name + ".publish.tmp")
    np.savez_compressed(vector_tmp, embeddings=vectors)
    ids_tmp.write_text(json.dumps(frame_ids, indent=2) + "\n", encoding="utf-8")
    try:
        loaded_vectors = np.load(vector_tmp)["embeddings"]
        loaded_ids = json.loads(ids_tmp.read_text(encoding="utf-8"))
        if len(loaded_vectors) != len(loaded_ids):
            raise RuntimeError("Temporary embedding pair is inconsistent")
        vector_tmp.replace(vector_path)
        ids_tmp.replace(ids_path)
    except Exception:
        vector_tmp.unlink(missing_ok=True)
        ids_tmp.unlink(missing_ok=True)
        raise


def _captioned_frame_ids(captions_path) -> set[str]:
    """Return frame_ids that already have a cached caption in captions.jsonl."""
    if not captions_path.exists():
        return set()
    from modules.captioning.validation import validate_caption

    manifest = JsonlManifest(captions_path)
    captioned: set[str] = set()
    for row in manifest.read_all():
        frame_id = row.get("frame_id")
        caption = row.get("caption")
        if frame_id and validate_caption(str(caption) if caption is not None else None).valid:
            captioned.add(str(frame_id))
    return captioned


def embed_frames(
    experiment: Experiment,
    batch_size: int = 32,
    force: bool = False,
    caption_missing: bool = False,
) -> int:
    """Embed extracted frames for every configured embedding model.

    Incremental per model: only frames not already embedded by a given model are
    processed and appended, so re-running after more frames are extracted embeds
    just the new ones. ``force`` discards existing embeddings and re-embeds
    everything. ``caption_missing`` lets vietnamese-embedding caption frames
    that have none yet (see the skip below) instead of passing over them.

    Every model always gets its own ``frames__<model>.npz`` /
    ``frame_ids__<model>.json`` — even with one model configured — so two
    separate embed-frames runs (e.g. different models on the same experiment)
    can never collide on the same output file.

    Returns the number of newly embedded (frame, model) pairs across all models.
    """
    specs = resolve_embedding_models(experiment.config.embedding_models)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install NumPy before embedding frames.") from exc

    frames_manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    embedding_manifest = JsonlManifest(experiment.run_dir / "manifests" / "embeddings.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    output_dir = experiment.run_dir / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [FrameRecord.from_dict(row) for row in frames_manifest.read_all(strict=True)]
    if not frames:
        raise ManifestError(f"Required frame manifest is missing or empty: {frames_manifest.path}")
    frames = [
        replace(
            frame,
            frame_path=str(require_experiment_frame_path(experiment, frame.frame_path)),
        )
        for frame in frames
    ]

    models = experiment.config.embedding_models
    specs_by_name = {spec.requested_name: spec for spec in specs}
    total_added = 0

    known_frame_ids = {frame.frame_id for frame in frames}
    manifest_by_model = {
        str(row.get("model_name")): row
        for row in embedding_manifest.read_all(strict=True)
        if row.get("model_name")
    }

    for model_index, model_name in enumerate(models, start=1):
        spec = specs_by_name[model_name]
        state.mark(model_name, "EMBED", "RUNNING")
        model_vectors_path = vectors_path(output_dir, model_name)
        model_frame_ids_path = frame_ids_path(output_dir, model_name)
        model_checkpoint_vectors_path = checkpoint_vectors_path(output_dir, model_name)
        model_checkpoint_ids_path = checkpoint_frame_ids_path(output_dir, model_name)

        existing_ids: list[str] = []
        existing_vectors = None
        if not force and model_vectors_path.exists() and model_frame_ids_path.exists():
            existing_ids = json.loads(model_frame_ids_path.read_text(encoding="utf-8"))
            existing_vectors = np.load(model_vectors_path)["embeddings"].astype("float32")

        # A prior interrupted run may have left mid-model progress behind —
        # fold it in as if it were already-embedded, so those frames aren't
        # redone. Discarded entirely when --force is passed, same as the
        # completed-model file above.
        checkpoint_ids, checkpoint_vectors = (
            _load_checkpoint(np, model_checkpoint_vectors_path, model_checkpoint_ids_path)
            if not force
            else ([], np.empty((0, 0), dtype="float32"))
        )
        if checkpoint_ids:
            LOGGER.info(
                "[%s] Resuming from checkpoint: %s frames already embedded this pass",
                model_name,
                len(checkpoint_ids),
            )

        already = set(existing_ids) | set(checkpoint_ids)
        new_frames = [frame for frame in frames if frame.frame_id not in already]

        # vietnamese-embedding embeds an existing caption rather than the
        # pixels; frames the VLM hasn't captioned yet have nothing to embed
        # and would otherwise trigger a live captioning call. Skip them here
        # so a captioning-only outage doesn't block embedding what's already
        # captioned — they'll be picked up on a later run once captioned.
        if model_name == "vietnamese-embedding" and not caption_missing:
            captioned_ids = _captioned_frame_ids(
                experiment.run_dir / "manifests" / "captions.jsonl"
            )
            skipped = [frame for frame in new_frames if frame.frame_id not in captioned_ids]
            new_frames = [frame for frame in new_frames if frame.frame_id in captioned_ids]
            if skipped:
                LOGGER.info(
                    "[%s] Skipping %s frames with no caption yet "
                    "(pass --caption-missing to caption them now)",
                    model_name,
                    len(skipped),
                )

        # Khong con frame moi de embed. Neu checkpoint dang giu vector chua
        # ghi vao file hoan chinh (vd: vietnamese-embedding bi chan boi thieu
        # caption) thi phai ghi ra truoc, tuyet doi khong duoc xoa truoc khi
        # ghi - xoa truoc se mat trang vector da tinh.
        if not new_frames:
            if not checkpoint_ids:
                LOGGER.info(
                    "[%s] All %s frames already embedded; nothing to do", model_name, len(frames)
                )
                state.mark(model_name, "EMBED", "COMPLETED")
                continue
            new_ids = checkpoint_ids
            new_vectors = np.asarray(checkpoint_vectors, dtype="float32")
        else:
            embedder = build_embedder(
                model_name=model_name,
                device=experiment.config.device,
                batch_size=batch_size,
                captions_path=experiment.run_dir / "manifests" / "captions.jsonl",
            )
            checkpoint_writer = _CheckpointWriter(
                np,
                model_checkpoint_vectors_path,
                model_checkpoint_ids_path,
                checkpoint_ids,
                checkpoint_vectors,
            )
            # Chia nho theo _EMBED_CHUNK_SIZE de RAM khong phinh to: vietnamese-embedding
            # giu ca future + caption cho tung frame dang xu ly, voi 60k frame trong 1
            # lan goi tung len ~4GB va lam doi GPU embedder khac chay cung.
            #
            # KHONG giu song song 1 ban fresh_ids/fresh_vectors trong RAM - checkpoint_writer
            # da la nguon su that duy nhat (flush xuong dia moi _CHECKPOINT_INTERVAL_SECONDS
            # hoac cuoi cung o day). Voi vai tram nghin frame, giu 2 ban trung lap se lam
            # RAM phinh gap doi khong can thiet.
            expected_count = len(checkpoint_ids)

            def _record_embedded_batch(
                batch_frames: list[FrameRecord], batch_vectors: list[list[float]]
            ) -> None:
                nonlocal expected_count
                checkpoint_writer.add_batch(batch_frames, batch_vectors)
                expected_count += len(batch_frames)

            chunk_total = math.ceil(len(new_frames) / _EMBED_CHUNK_SIZE)
            for chunk_index, start in enumerate(
                range(0, len(new_frames), _EMBED_CHUNK_SIZE), start=1
            ):
                chunk = new_frames[start : start + _EMBED_CHUNK_SIZE]
                LOGGER.info(
                    "[model %s/%s] [batch %s/%s] Embedding model=%s frames=%s",
                    model_index,
                    len(models),
                    chunk_index,
                    chunk_total,
                    model_name,
                    len(chunk),
                )
                before = expected_count
                returned_vectors = embedder.embed_images(chunk, on_batch=_record_embedded_batch)
                if expected_count == before and returned_vectors:
                    if len(returned_vectors) != len(chunk):
                        raise EmbeddingError(
                            f"Embedder '{model_name}' returned {len(returned_vectors)} "
                            f"vectors for {len(chunk)} frames without reporting frame ids."
                        )
                    _record_embedded_batch(chunk, returned_vectors)

            # Ep flush lan cuoi (add_batch chi flush moi _CHECKPOINT_INTERVAL_SECONDS,
            # nen phan chua flush cua batch cuoi van con nam trong RAM cua
            # checkpoint_writer, chua ra dia) roi doc lai tu dia lam ket qua cuoi cung.
            checkpoint_writer.flush()
            new_ids, new_vectors = _load_checkpoint(
                np, model_checkpoint_vectors_path, model_checkpoint_ids_path
            )

        # Moi frame trong new_frames co the that bai het (vd: toan bo con lai
        # deu bi VLM tu choi vi Han/CJK) - new_vectors khi do la mang rong 1
        # chieu, khong ghep duoc voi existing_vectors 2 chieu qua concatenate.
        # Khong co gi moi de ghi thi giu nguyen file cu, khoi dung toi no.
        if not new_ids:
            LOGGER.info(
                "[%s] No frame captioned successfully this pass; leaving existing file untouched",
                model_name,
            )
            continue

        if existing_vectors is not None and len(existing_vectors):
            vectors = np.concatenate([existing_vectors, new_vectors], axis=0)
            frame_ids = existing_ids + new_ids
        else:
            vectors = new_vectors
            frame_ids = new_ids

        _validate_artifact(np, vectors, frame_ids, known_frame_ids)
        _publish_artifact(np, model_vectors_path, model_frame_ids_path, vectors, frame_ids)
        _discard_checkpoint(model_checkpoint_vectors_path, model_checkpoint_ids_path)
        manifest_by_model[model_name] = {
            "embedding_path": str(model_vectors_path),
            "frame_ids_path": str(model_frame_ids_path),
            "added": len(new_ids),
            "total": len(frame_ids),
            "dimension": int(vectors.shape[1]),
            "coverage_ratio": len(set(frame_ids) & known_frame_ids) / len(known_frame_ids),
            "model_name": model_name,
            "requested_name": spec.requested_name,
            "backend": spec.backend,
            "resolved_model_id": spec.resolved_model_id,
            "revision": spec.revision,
            "preprocessing": spec.preprocessing,
        }
        state.mark(model_name, "EMBED", "COMPLETED")
        LOGGER.info(
            "[%s] Embedded frames added=%s total=%s path=%s",
            model_name,
            len(new_ids),
            len(frame_ids),
            model_vectors_path,
        )
        total_added += len(new_ids)

    embedding_manifest.replace_all(manifest_by_model.values(), unique_key="model_name")
    state.mark("frames", "EMBED", "COMPLETED")
    return total_added
