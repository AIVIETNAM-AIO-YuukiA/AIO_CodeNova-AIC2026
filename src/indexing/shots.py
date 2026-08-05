"""Shot detection pipeline stage."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from core.logging import get_logger

from config.settings import Experiment
from core.types import ShotRecord, VideoRecord
from indexing.manifest import JsonlManifest
from indexing.state import JobState
from video.shots import TransNetV2ShotDetector, decode_video

LOGGER = get_logger(__name__)


def detect_shots(
    experiment: Experiment,
    weights_path: Path,
    module_dir: Path | None = None,
    force: bool = False,
) -> int:
    """Detect shots for all discovered videos and save them to ``shots.jsonl``."""
    videos_manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
    shots_manifest = JsonlManifest(experiment.run_dir / "manifests" / "shots.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    detector = TransNetV2ShotDetector(
        weights_path=weights_path,
        module_dir=module_dir,
        device=experiment.config.device,
    )
    existing_by_video = _existing_shot_video_ids(shots_manifest)
    recorded = 0

    pending = []
    for row in videos_manifest.read_all():
        video = VideoRecord.from_dict(row)
        if (
            state.should_skip(video.video_id, "SHOT_DETECT", force=force)
            and video.video_id in existing_by_video
        ):
            LOGGER.info("Skipping shot detection video_id=%s", video.video_id)
            continue
        pending.append(video)

    if not pending:
        LOGGER.info("No videos need shot detection")
        return 0

    # Decoding is CPU-only and inference is GPU-only, so decode the next video
    # while the current one is on the GPU. One worker is enough to hide it.
    with ThreadPoolExecutor(max_workers=1) as decoder:
        upcoming = decoder.submit(decode_video, pending[0].path)
        for index, video in enumerate(pending):
            try:
                decoded = upcoming.result()
            except Exception as exc:
                LOGGER.exception("Video decode failed video_id=%s", video.video_id)
                state.mark(video.video_id, "SHOT_DETECT", "FAILED", error=str(exc))
                decoded = None
            if index + 1 < len(pending):
                upcoming = decoder.submit(decode_video, pending[index + 1].path)
            if decoded is None:
                continue
            try:
                shots = detector.detect_decoded(video, decoded)
                shots_manifest.extend(shot.to_dict() for shot in shots)
                state.mark(video.video_id, "SHOT_DETECT", "COMPLETED")
                recorded += len(shots)
                LOGGER.info("Detected shots video_id=%s count=%s", video.video_id, len(shots))
            except Exception as exc:
                LOGGER.exception("Shot detection failed video_id=%s", video.video_id)
                state.mark(video.video_id, "SHOT_DETECT", "FAILED", error=str(exc))
    return recorded


def load_shots_by_video(experiment: Experiment) -> dict[str, list[ShotRecord]]:
    """Load shot records grouped by video ID."""
    manifest = JsonlManifest(experiment.run_dir / "manifests" / "shots.jsonl")
    grouped: dict[str, list[ShotRecord]] = defaultdict(list)
    for row in manifest.read_all():
        shot = ShotRecord.from_dict(row)
        grouped[shot.video_id].append(shot)
    return grouped


def _existing_shot_video_ids(manifest: JsonlManifest) -> set[str]:
    return {str(row["video_id"]) for row in manifest.read_all() if "video_id" in row}
