"""Frame extraction pipeline stage."""

from __future__ import annotations

import logging

from config.settings import Experiment
from core.types import VideoRecord
from pipeline.manifest import JsonlManifest
from pipeline.shots import load_shots_by_video
from pipeline.state import JobState
from video.frames import OpenCVFrameExtractor

LOGGER = logging.getLogger(__name__)


def extract_frames(experiment: Experiment, force: bool = False) -> int:
    """Extract keyframes for detected shots and save ``frames.jsonl``."""
    videos_manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
    frames_manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    shots_by_video = load_shots_by_video(experiment)
    extractor = OpenCVFrameExtractor(
        output_dir=experiment.run_dir / "frames",
        keyframes_per_shot=experiment.config.keyframes_per_shot,
    )
    existing_video_ids = {
        str(row["video_id"]) for row in frames_manifest.read_all() if "video_id" in row
    }
    recorded = 0

    for row in videos_manifest.read_all():
        video = VideoRecord.from_dict(row)
        if (
            state.should_skip(video.video_id, "FRAME_EXTRACT", force=force)
            and video.video_id in existing_video_ids
        ):
            LOGGER.info("Skipping frame extraction video_id=%s", video.video_id)
            continue
        shots = shots_by_video.get(video.video_id, [])
        if not shots:
            LOGGER.warning("No shots found for video_id=%s; skipping frames", video.video_id)
            continue
        try:
            frames = extractor.extract(video, shots)
            frames_manifest.extend(frame.to_dict() for frame in frames)
            state.mark(video.video_id, "FRAME_EXTRACT", "COMPLETED")
            recorded += len(frames)
            LOGGER.info("Extracted frames video_id=%s count=%s", video.video_id, len(frames))
        except Exception as exc:
            LOGGER.exception("Frame extraction failed video_id=%s", video.video_id)
            state.mark(video.video_id, "FRAME_EXTRACT", "FAILED", error=str(exc))
    return recorded
