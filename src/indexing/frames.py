"""Frame extraction pipeline stage."""

from __future__ import annotations

from core.logging import get_logger
from pathlib import Path
import shutil
import uuid

from config.settings import Experiment
from core.paths import canonical_frame_path
from core.types import FrameRecord, VideoRecord
from indexing.manifest import JsonlManifest, ManifestError
from indexing.partitions import PartitionStore
from indexing.shots import load_shots_by_video
from indexing.state import JobState
from video.frames import FFmpegFrameExtractor

LOGGER = get_logger(__name__)


def extract_frames(experiment: Experiment, force: bool = False) -> int:
    """Extract keyframes for detected shots and save ``frames.jsonl``."""
    videos_manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
    frames_manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    shots_by_video = load_shots_by_video(experiment)
    frames_root = experiment.run_dir / "frames"
    partition_store = PartitionStore(experiment.run_dir / "manifests" / "partitions" / "frames")
    recorded = 0

    video_rows = videos_manifest.read_all(strict=True)
    if not video_rows:
        raise ManifestError(f"Required video manifest is missing or empty: {videos_manifest.path}")
    for position, row in enumerate(video_rows, start=1):
        video = VideoRecord.from_dict(row)
        if state.should_skip(
            video.video_id, "FRAME_EXTRACT", force=force
        ) and partition_store.exists(video.video_id):
            LOGGER.info("Skipping frame extraction video_id=%s", video.video_id)
            continue
        shots = shots_by_video.get(video.video_id, [])
        if not shots:
            LOGGER.warning("No shots found for video_id=%s; skipping frames", video.video_id)
            state.mark(
                video.video_id,
                "FRAME_EXTRACT",
                "BLOCKED",
                error="No validated shot records",
            )
            continue
        try:
            state.mark(video.video_id, "FRAME_EXTRACT", "RUNNING")
            staging_root = frames_root / ".staging" / f"{video.video_id}_{uuid.uuid4().hex[:8]}"
            extractor = FFmpegFrameExtractor(
                output_dir=staging_root,
                keyframe_percentiles=experiment.config.keyframe_percentiles,
            )
            frames = extractor.extract(video, shots)
            if not frames:
                raise RuntimeError(f"No frames extracted for video_id={video.video_id}")
            frame_ids = [frame.frame_id for frame in frames]
            known_shot_ids = {shot.shot_id for shot in shots}
            if len(frame_ids) != len(set(frame_ids)):
                raise RuntimeError(f"Duplicate frame IDs for video_id={video.video_id}")
            for frame in frames:
                if frame.video_id != video.video_id or frame.shot_id not in known_shot_ids:
                    raise RuntimeError(f"Invalid frame references frame_id={frame.frame_id}")
                if not Path(frame.frame_path).exists():
                    raise RuntimeError(f"Missing extracted frame file={frame.frame_path}")

            generated_dir = staging_root / video.video_id
            final_dir = frames_root / video.video_id
            backup_dir = frames_root / f".{video.video_id}.previous"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            if final_dir.exists():
                final_dir.replace(backup_dir)
            generated_dir.replace(final_dir)
            published_frames = [
                FrameRecord(
                    frame_id=frame.frame_id,
                    video_id=frame.video_id,
                    shot_id=frame.shot_id,
                    frame_path=canonical_frame_path(
                        experiment, final_dir / Path(frame.frame_path).name
                    ),
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                )
                for frame in frames
            ]
            partition_store.write(
                video.video_id,
                (frame.to_dict() for frame in published_frames),
                partition_key="video_id",
                unique_key="frame_id",
            )
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.rmtree(staging_root, ignore_errors=True)
            state.mark(video.video_id, "FRAME_EXTRACT", "COMPLETED")
            recorded += len(frames)
            LOGGER.info(
                "[video %s/%s] Extracted frames video_id=%s count=%s",
                position,
                len(video_rows),
                video.video_id,
                len(frames),
            )
        except Exception as exc:
            LOGGER.exception("Frame extraction failed video_id=%s", video.video_id)
            staging = locals().get("staging_root")
            if isinstance(staging, Path):
                shutil.rmtree(staging, ignore_errors=True)
            backup = frames_root / f".{video.video_id}.previous"
            final = frames_root / video.video_id
            if backup.exists():
                if final.exists():
                    shutil.rmtree(final)
                backup.replace(final)
            state.mark(video.video_id, "FRAME_EXTRACT", "FAILED", error=str(exc))
    partition_store.consolidate(frames_manifest.path, unique_key="frame_id")
    return recorded
