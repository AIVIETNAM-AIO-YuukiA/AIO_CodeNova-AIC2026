"""Vector index build and search helpers."""

from __future__ import annotations

from pathlib import Path
import json

from config.settings import Experiment
from core.types import FrameRecord, SearchResult, VideoRecord
from embedding.clip_model import TransformersClipEmbedder
from index.faiss_index import FaissVectorIndex
from pipeline.manifest import JsonlManifest
from retrieval.query_processor import get_query_processor


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
    # Check if we should use GPU based on experiment device setting
    use_gpu = experiment.config.device != "cpu"
    index = FaissVectorIndex(
        index_path=experiment.run_dir / "index" / "frames.faiss",
        mapping_path=experiment.run_dir / "index" / "frame_ids.json",
        use_gpu=use_gpu,
        require_gpu=False,
    )
    processor = get_query_processor()
    processed = processor.process(query)
    raw_results = index.search(embedder.embed_text(processed.visual_prompt), top_k=top_k)
    frame_lookup = _load_frame_lookup(experiment)
    video_lookup = _load_video_lookup(experiment)
    hydrated = []
    for result in raw_results:
        frame = frame_lookup.get(result.frame_id)
        if frame is None:
            hydrated.append(result)
            continue
        video = video_lookup.get(frame.video_id)
        hydrated.append(
            SearchResult(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                score=result.score,
                frame_path=frame.frame_path,
                video_path=video.path if video else None,
                video_name=Path(video.path).name if video else None,
                shot_id=frame.shot_id,
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
            )
        )
    return hydrated


def _load_frame_lookup(experiment: Experiment) -> dict[str, FrameRecord]:
    manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    frames = [FrameRecord.from_dict(row) for row in manifest.read_all()]
    return {frame.frame_id: frame for frame in frames}


def _load_video_lookup(experiment: Experiment) -> dict[str, VideoRecord]:
    manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
    videos = [VideoRecord.from_dict(row) for row in manifest.read_all()]
    return {video.video_id: video for video in videos}
