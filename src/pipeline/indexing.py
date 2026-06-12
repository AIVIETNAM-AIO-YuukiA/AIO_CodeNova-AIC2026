"""Vector index build and search helpers."""

from __future__ import annotations

import json

from config.settings import Experiment
from core.types import FrameRecord, SearchResult
from embedding.clip_model import TransformersClipEmbedder
from index.faiss_index import FaissVectorIndex
from pipeline.manifest import JsonlManifest


def build_index(experiment: Experiment, force: bool = False) -> int:
    """Build a FAISS index from saved frame embeddings."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install NumPy before building the index.") from exc

    vectors_path = experiment.run_dir / "embeddings" / "frames.npz"
    frame_ids_path = experiment.run_dir / "embeddings" / "frame_ids.json"
    index_path = experiment.run_dir / "index" / "frames.faiss"
    mapping_path = experiment.run_dir / "index" / "frame_ids.json"
    if index_path.exists() and mapping_path.exists() and not force:
        return 0

    vectors = np.load(vectors_path)["embeddings"].astype("float32")
    frame_ids = json.loads(frame_ids_path.read_text(encoding="utf-8"))
    index = FaissVectorIndex(index_path=index_path, mapping_path=mapping_path)
    index.build(vectors.tolist(), frame_ids)
    return len(frame_ids)


def search_index(experiment: Experiment, query: str, top_k: int) -> list[SearchResult]:
    """Search the built index and hydrate frame/video metadata."""
    embedder = TransformersClipEmbedder(
        model_name=experiment.config.clip_model,
        device=experiment.config.device,
    )
    index = FaissVectorIndex(
        index_path=experiment.run_dir / "index" / "frames.faiss",
        mapping_path=experiment.run_dir / "index" / "frame_ids.json",
    )
    raw_results = index.search(embedder.embed_text(query), top_k=top_k)
    frame_lookup = _load_frame_lookup(experiment)
    hydrated = []
    for result in raw_results:
        frame = frame_lookup.get(result.frame_id)
        if frame is None:
            hydrated.append(result)
            continue
        hydrated.append(
            SearchResult(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                score=result.score,
                frame_path=frame.frame_path,
                timestamp_sec=frame.timestamp_sec,
            )
        )
    return hydrated


def _load_frame_lookup(experiment: Experiment) -> dict[str, FrameRecord]:
    manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    frames = [FrameRecord.from_dict(row) for row in manifest.read_all()]
    return {frame.frame_id: frame for frame in frames}
