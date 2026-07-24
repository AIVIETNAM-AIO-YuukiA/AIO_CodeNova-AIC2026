"""Embedding pipeline stage (incremental, multi-model)."""

from __future__ import annotations

import json
import time
from core.logging import get_logger

from config.settings import Experiment
from core.types import FrameRecord
from modules.embedding import build_embedder
from indexing.manifest import JsonlManifest
from indexing.state import JobState

LOGGER = get_logger(__name__)

# How often (seconds) an in-progress model flushes its checkpoint to disk.
# embed_images() can run for hours over hundreds of thousands of frames;
# checkpointing lets a killed/interrupted run resume mid-model instead of
# redoing all of it, at the cost of a small periodic disk write.
_CHECKPOINT_INTERVAL_SECONDS = 30.0


def _vectors_path(output_dir, model_name: str, single_model: bool):
    if single_model:
        return output_dir / "frames.npz"
    return output_dir / f"frames__{model_name}.npz"


def _checkpoint_paths(vectors_path, frame_ids_path):
    return vectors_path.with_suffix(".checkpoint.npz"), frame_ids_path.with_suffix(
        ".checkpoint.json"
    )


class _CheckpointWriter:
    """Buffers embedded (frame_id, vector) pairs and periodically flushes them
    to a ``*.checkpoint.npz`` / ``*.checkpoint.json`` pair, so a run killed
    mid-model can resume from the last flush instead of from scratch."""

    def __init__(self, np, checkpoint_vectors_path, checkpoint_ids_path, prior_ids, prior_vectors):
        self._np = np
        self._vectors_path = checkpoint_vectors_path
        self._ids_path = checkpoint_ids_path
        self._ids: list[str] = list(prior_ids)
        self._vectors: list = list(prior_vectors)
        self._last_flush = time.monotonic()

    def add_batch(self, frames: list[FrameRecord], vectors: list[list[float]]) -> None:
        self._ids.extend(frame.frame_id for frame in frames)
        self._vectors.extend(vectors)
        now = time.monotonic()
        if now - self._last_flush >= _CHECKPOINT_INTERVAL_SECONDS:
            self.flush()
            self._last_flush = now

    def flush(self) -> None:
        if not self._ids:
            return
        array = self._np.asarray(self._vectors, dtype="float32")
        self._np.savez_compressed(self._vectors_path, embeddings=array)
        self._ids_path.write_text(json.dumps(self._ids) + "\n", encoding="utf-8")


def _load_checkpoint(np, checkpoint_vectors_path, checkpoint_ids_path):
    """Return (ids, vectors) from a prior interrupted run's checkpoint, or empty lists."""
    if not (checkpoint_vectors_path.exists() and checkpoint_ids_path.exists()):
        return [], []
    ids = json.loads(checkpoint_ids_path.read_text(encoding="utf-8"))
    vectors = np.load(checkpoint_vectors_path)["embeddings"].astype("float32").tolist()
    return ids, vectors


def _discard_checkpoint(checkpoint_vectors_path, checkpoint_ids_path) -> None:
    checkpoint_vectors_path.unlink(missing_ok=True)
    checkpoint_ids_path.unlink(missing_ok=True)


def embed_frames(experiment: Experiment, batch_size: int = 32, force: bool = False) -> int:
    """Embed extracted frames for every configured embedding model.

    Incremental per model: only frames not already embedded by a given model are
    processed and appended, so re-running after more frames are extracted embeds
    just the new ones. ``force`` discards existing embeddings and re-embeds
    everything. Each model's vectors are stored separately (different backends
    produce different dimensionalities) but share one ``frame_ids.json`` ordering
    since every model is run over the same frame set.

    With a single configured model the legacy file names are kept
    (``frames.npz`` / ``frame_ids.json``) so existing runs stay compatible.
    With multiple models, each gets its own ``frames__<model>.npz``.

    Returns the number of newly embedded (frame, model) pairs across all models.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install NumPy before embedding frames.") from exc

    frames_manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    embedding_manifest = JsonlManifest(experiment.run_dir / "manifests" / "embeddings.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    output_dir = experiment.run_dir / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [FrameRecord.from_dict(row) for row in frames_manifest.read_all()]
    if not frames:
        LOGGER.warning("No frames found to embed")
        return 0

    models = experiment.config.embedding_models
    single_model = len(models) == 1
    total_added = 0

    for model_name in models:
        vectors_path = _vectors_path(output_dir, model_name, single_model)
        frame_ids_path = output_dir / (
            "frame_ids.json" if single_model else f"frame_ids__{model_name}.json"
        )
        checkpoint_vectors_path, checkpoint_ids_path = _checkpoint_paths(
            vectors_path, frame_ids_path
        )

        existing_ids: list[str] = []
        existing_vectors = None
        if not force and vectors_path.exists() and frame_ids_path.exists():
            existing_ids = json.loads(frame_ids_path.read_text(encoding="utf-8"))
            existing_vectors = np.load(vectors_path)["embeddings"].astype("float32")

        # A prior interrupted run may have left mid-model progress behind —
        # fold it in as if it were already-embedded, so those frames aren't
        # redone. Discarded entirely when --force is passed, same as the
        # completed-model file above.
        checkpoint_ids, checkpoint_vectors = (
            _load_checkpoint(np, checkpoint_vectors_path, checkpoint_ids_path)
            if not force
            else ([], [])
        )
        if checkpoint_ids:
            LOGGER.info(
                "[%s] Resuming from checkpoint: %s frames already embedded this pass",
                model_name,
                len(checkpoint_ids),
            )

        already = set(existing_ids) | set(checkpoint_ids)
        new_frames = [frame for frame in frames if frame.frame_id not in already]
        if not new_frames:
            _discard_checkpoint(checkpoint_vectors_path, checkpoint_ids_path)
            LOGGER.info(
                "[%s] All %s frames already embedded; nothing to do", model_name, len(frames)
            )
            continue

        embedder = build_embedder(
            model_name=model_name,
            device=experiment.config.device,
            batch_size=batch_size,
            captions_path=experiment.run_dir / "manifests" / "captions.jsonl",
        )
        checkpoint_writer = _CheckpointWriter(
            np, checkpoint_vectors_path, checkpoint_ids_path, checkpoint_ids, checkpoint_vectors
        )
        fresh_vectors = embedder.embed_images(new_frames, on_batch=checkpoint_writer.add_batch)

        # new_frames already excludes anything the checkpoint covered, so the
        # final per-model result is checkpoint carry-over + this pass's
        # output, in that order (matches how the checkpoint buffer accumulated).
        new_ids = checkpoint_ids + [frame.frame_id for frame in new_frames]
        new_vectors = np.asarray(checkpoint_vectors + list(fresh_vectors), dtype="float32")

        if existing_vectors is not None and len(existing_vectors):
            vectors = np.concatenate([existing_vectors, new_vectors], axis=0)
            frame_ids = existing_ids + new_ids
        else:
            vectors = new_vectors
            frame_ids = new_ids

        np.savez_compressed(vectors_path, embeddings=vectors)
        frame_ids_path.write_text(json.dumps(frame_ids, indent=2) + "\n", encoding="utf-8")
        _discard_checkpoint(checkpoint_vectors_path, checkpoint_ids_path)
        embedding_manifest.append(
            {
                "embedding_path": str(vectors_path),
                "frame_ids_path": str(frame_ids_path),
                "added": len(new_ids),
                "total": len(frame_ids),
                "model_name": model_name,
            }
        )
        LOGGER.info(
            "[%s] Embedded frames added=%s total=%s path=%s",
            model_name,
            len(new_ids),
            len(frame_ids),
            vectors_path,
        )
        total_added += len(new_ids)

    state.mark("frames", "EMBED", "COMPLETED")
    return total_added
