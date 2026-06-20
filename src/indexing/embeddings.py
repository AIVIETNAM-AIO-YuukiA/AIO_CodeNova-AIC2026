"""Embedding pipeline stage (incremental)."""

from __future__ import annotations

import json
from core.logging import get_logger

from config.settings import Experiment
from core.types import FrameRecord
from modules.embedding import build_embedder
from indexing.manifest import JsonlManifest
from indexing.state import JobState

LOGGER = get_logger(__name__)


def embed_frames(experiment: Experiment, batch_size: int = 32, force: bool = False) -> int:
    """Embed extracted frames into ``embeddings/frames.npz`` plus metadata.

    Incremental: only frames not already embedded are processed and their vectors
    are appended, so re-running after more frames are extracted embeds just the
    new ones. ``force`` discards existing embeddings and re-embeds everything.
    The row order of ``frames.npz`` matches ``frame_ids.json`` one-to-one.

    Returns the number of newly embedded frames.
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
    vectors_path = output_dir / "frames.npz"
    frame_ids_path = output_dir / "frame_ids.json"

    frames = [FrameRecord.from_dict(row) for row in frames_manifest.read_all()]
    if not frames:
        LOGGER.warning("No frames found to embed")
        return 0

    existing_ids: list[str] = []
    existing_vectors = None
    if not force and vectors_path.exists() and frame_ids_path.exists():
        existing_ids = json.loads(frame_ids_path.read_text(encoding="utf-8"))
        existing_vectors = np.load(vectors_path)["embeddings"].astype("float32")

    already = set(existing_ids)
    new_frames = [frame for frame in frames if frame.frame_id not in already]
    if not new_frames:
        LOGGER.info("All %s frames already embedded; nothing to do", len(frames))
        return 0

    embedder = build_embedder(
        model_name=experiment.config.embedding_model,
        device=experiment.config.device,
        batch_size=batch_size,
    )
    new_vectors = np.asarray(embedder.embed_images(new_frames), dtype="float32")
    new_ids = [frame.frame_id for frame in new_frames]

    if existing_vectors is not None and len(existing_vectors):
        vectors = np.concatenate([existing_vectors, new_vectors], axis=0)
        frame_ids = existing_ids + new_ids
    else:
        vectors = new_vectors
        frame_ids = new_ids

    np.savez_compressed(vectors_path, embeddings=vectors)
    frame_ids_path.write_text(json.dumps(frame_ids, indent=2) + "\n", encoding="utf-8")
    embedding_manifest.append(
        {
            "embedding_path": str(vectors_path),
            "frame_ids_path": str(frame_ids_path),
            "added": len(new_ids),
            "total": len(frame_ids),
            "model_name": experiment.config.embedding_model,
        }
    )
    state.mark("frames", "EMBED", "COMPLETED")
    LOGGER.info(
        "Embedded frames added=%s total=%s path=%s", len(new_ids), len(frame_ids), vectors_path
    )
    return len(new_ids)
