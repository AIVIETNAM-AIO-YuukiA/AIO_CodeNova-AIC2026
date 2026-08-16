"""Ingestion orchestration."""

from __future__ import annotations

from pathlib import Path
from core.logging import get_logger

from config.settings import Experiment
from indexing.manifest import JsonlManifest
from indexing.state import JobState
from indexing.partitions import PartitionStore
from video.discovery import discover_video_paths, inspect_video

LOGGER = get_logger(__name__)


def ingest_videos(experiment: Experiment, input_dir: Path, force: bool = False) -> int:
    """Discover videos and persist them to the experiment manifest.

    Returns the number of newly recorded videos.
    """
    manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    partition_store = PartitionStore(experiment.run_dir / "manifests" / "partitions" / "videos")
    recorded = 0

    LOGGER.info("Discovering videos in %s", input_dir)
    paths = list(discover_video_paths(input_dir))
    if not paths:
        raise RuntimeError(f"No supported videos found under {input_dir}")
    for position, path in enumerate(paths, start=1):
        item_id = str(path)
        try:
            video = inspect_video(path)
        except Exception as exc:
            LOGGER.exception("[video %s/%s] Discovery failed path=%s", position, len(paths), path)
            state.mark(item_id, "DISCOVER", "FAILED", error=str(exc))
            continue
        if state.should_skip(video.video_id, "DISCOVER", force=force) and partition_store.exists(
            video.video_id
        ):
            LOGGER.info(
                "[video %s/%s] Skipping video_id=%s path=%s",
                position,
                len(paths),
                video.video_id,
                video.path,
            )
            continue
        try:
            state.mark(video.video_id, "DISCOVER", "RUNNING")
            partition_store.write(
                video.video_id,
                [video.to_dict()],
                partition_key="video_id",
                unique_key="video_id",
            )
            state.mark(video.video_id, "DISCOVER", "COMPLETED")
            recorded += 1
            LOGGER.info(
                "[video %s/%s] Recorded video_id=%s path=%s",
                position,
                len(paths),
                video.video_id,
                video.path,
            )
        except Exception as exc:
            LOGGER.exception("Failed to record video_id=%s path=%s", video.video_id, video.path)
            state.mark(video.video_id, "DISCOVER", "FAILED", error=str(exc))

    partition_store.consolidate(manifest.path, unique_key="video_id")
    LOGGER.info("Discovery complete recorded=%s total=%s", recorded, len(paths))
    return recorded
