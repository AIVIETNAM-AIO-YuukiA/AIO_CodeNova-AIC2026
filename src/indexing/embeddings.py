"""CLIP embedding pipeline stage."""

from __future__ import annotations

import json
from core.logging import get_logger

from config.settings import Experiment
from core.types import FrameRecord
from modules.embedding import TransformersClipEmbedder
from indexing.manifest import JsonlManifest
from indexing.state import JobState

LOGGER = get_logger(__name__)


def embed_frames(experiment: Experiment, batch_size: int = 32, force: bool = False) -> int:
    """Embed extracted frames and save ``embeddings/frames.npz`` plus metadata."""
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

    if vectors_path.exists() and frame_ids_path.exists() and not force:
        LOGGER.info("Skipping embeddings because %s already exists", vectors_path)
        return 0

    frames = [FrameRecord.from_dict(row) for row in frames_manifest.read_all()]
    if not frames:
        LOGGER.warning("No frames found to embed")
        return 0

    embedder = TransformersClipEmbedder(
        model_name=experiment.config.clip_model,
        device=experiment.config.device,
        batch_size=batch_size,
    )
    vectors = embedder.embed_images(frames)
    frame_ids = [frame.frame_id for frame in frames]
    np.savez_compressed(vectors_path, embeddings=np.asarray(vectors, dtype="float32"))
    frame_ids_path.write_text(json.dumps(frame_ids, indent=2) + "\n", encoding="utf-8")
    embedding_manifest.append(
        {
            "embedding_path": str(vectors_path),
            "frame_ids_path": str(frame_ids_path),
            "count": len(frame_ids),
            "model_name": experiment.config.clip_model,
        }
    )
    state.mark("frames", "EMBED", "COMPLETED")
    LOGGER.info("Embedded frames count=%s path=%s", len(frame_ids), vectors_path)
    return len(frame_ids)
