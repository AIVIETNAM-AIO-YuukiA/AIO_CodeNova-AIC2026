"""Ingestion orchestration."""

from __future__ import annotations

from pathlib import Path
from core.logging import get_logger

from config.settings import Experiment
from indexing.manifest import JsonlManifest
from indexing.state import JobState
from video.discovery import discover_videos

LOGGER = get_logger(__name__)


def ingest_videos(experiment: Experiment, input_dir: Path, force: bool = False) -> int:
    """Discover videos and persist them to the experiment manifest.

    Returns the number of newly recorded videos.
    """
    manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    known_ids = manifest.ids("video_id")
    recorded = 0

    LOGGER.info("Discovering videos in %s", input_dir)
    for video in discover_videos(input_dir):
        if (
            state.should_skip(video.video_id, "DISCOVER", force=force)
            and video.video_id in known_ids
        ):
            LOGGER.info(
                "Skipping already discovered video_id=%s path=%s", video.video_id, video.path
            )
            continue
        try:
            manifest.append(video.to_dict())
            state.mark(video.video_id, "DISCOVER", "COMPLETED")
            recorded += 1
            LOGGER.info("Recorded video_id=%s path=%s", video.video_id, video.path)
        except Exception as exc:
            LOGGER.exception("Failed to record video_id=%s path=%s", video.video_id, video.path)
            state.mark(video.video_id, "DISCOVER", "FAILED", error=str(exc))

    LOGGER.info("Discovery complete recorded=%s", recorded)
    return recorded
