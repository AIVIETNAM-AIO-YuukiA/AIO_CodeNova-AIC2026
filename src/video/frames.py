"""Keyframe extraction interfaces."""

from __future__ import annotations

from pathlib import Path
import subprocess

from core.errors import FrameExtractionError
from core.types import FrameRecord, ShotRecord, VideoRecord


class FrameExtractor:
    """Interface for extracting representative frames from shots."""

    def extract(self, video: VideoRecord, shots: list[ShotRecord]) -> list[FrameRecord]:
        """Extract keyframes for a video's shots."""
        raise NotImplementedError


DEFAULT_KEYFRAME_PERCENTILES = (0.15, 0.5, 0.85)


class FFmpegFrameExtractor(FrameExtractor):
    """ffmpeg-backed keyframe extractor sampling shots at fixed percentiles.

    Unlike per-frame random seeking, this decodes each video sequentially exactly
    once and emits only the requested frame indices via the ffmpeg ``select``
    filter. Frame positions match the percentile sampling exactly; only the read
    strategy changes, so output is identical to a seek-based extractor.
    """

    def __init__(
        self,
        output_dir: Path,
        keyframe_percentiles: tuple[float, ...] = DEFAULT_KEYFRAME_PERCENTILES,
    ) -> None:
        self.output_dir = output_dir
        self.keyframe_percentiles = tuple(keyframe_percentiles)

    def extract(self, video: VideoRecord, shots: list[ShotRecord]) -> list[FrameRecord]:
        """Extract percentile keyframes for every shot in one ffmpeg pass."""
        fps = probe_fps(video.path)
        video_dir = self.output_dir / video.video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        # Collect every wanted frame index across all shots, sorted & unique.
        # Map each frame index back to the shot it belongs to for the FrameRecord.
        index_to_shot: dict[int, ShotRecord] = {}
        for shot in shots:
            for frame_index in sample_frame_indices(
                shot.start_frame, shot.end_frame, self.keyframe_percentiles
            ):
                index_to_shot.setdefault(frame_index, shot)
        wanted = sorted(index_to_shot)
        if not wanted:
            return []

        _run_ffmpeg_select(video.path, wanted, video_dir / "_kf_%06d.jpg")

        records: list[FrameRecord] = []
        for ordinal, frame_index in enumerate(wanted, start=1):
            produced = video_dir / f"_kf_{ordinal:06d}.jpg"
            if not produced.exists():
                raise FrameExtractionError(
                    f"ffmpeg did not emit frame={frame_index} video={video.path}"
                )
            shot = index_to_shot[frame_index]
            frame_id = f"{shot.shot_id}_f{frame_index:08d}"
            frame_path = video_dir / f"{frame_id}.jpg"
            produced.replace(frame_path)
            records.append(
                FrameRecord(
                    frame_id=frame_id,
                    video_id=video.video_id,
                    shot_id=shot.shot_id,
                    frame_path=str(frame_path),
                    frame_index=frame_index,
                    timestamp_sec=(frame_index / fps) if fps else None,
                )
            )
        return records


def _run_ffmpeg_select(video_path: str, frame_indices: list[int], output_pattern: Path) -> None:
    """Decode ``video_path`` once and write the given frame indices as JPEGs.

    The ``select`` filter keeps only frames whose 0-based index is in the set;
    ``-vsync 0`` (passthrough) makes ffmpeg emit exactly the selected frames so
    the output numbering matches the sorted index order one-to-one.
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise FrameExtractionError("Install imageio-ffmpeg before extracting frames.") from exc

    select_expr = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "0",
        "-q:v",
        "2",
        "-y",
        str(output_pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise FrameExtractionError(f"ffmpeg failed for {video_path}: {result.stderr.strip()[:500]}")


def probe_fps(video_path: str) -> float:
    """Return the video frame rate, or 0.0 if it cannot be read."""
    try:
        import cv2
    except ImportError as exc:
        raise FrameExtractionError("Install opencv-python before extracting frames.") from exc

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FrameExtractionError(f"Cannot open video: {video_path}")
    try:
        return capture.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        capture.release()


def sample_frame_indices(
    start_frame: int, end_frame: int, percentiles: tuple[float, ...]
) -> list[int]:
    """Return representative frame indices at the given percentiles of a shot.

    Each index is ``start_frame + round(span * p)`` for percentile ``p``.
    Duplicate indices (e.g. for very short shots) are collapsed so a 1-frame
    shot yields a single keyframe instead of repeats.
    """
    if not percentiles:
        raise FrameExtractionError("keyframe_percentiles must not be empty.")
    if end_frame < start_frame:
        raise FrameExtractionError("Shot end_frame cannot be smaller than start_frame.")
    span = end_frame - start_frame
    indices: list[int] = []
    for percentile in percentiles:
        if not 0.0 <= percentile <= 1.0:
            raise FrameExtractionError(f"Percentile {percentile} is outside [0, 1].")
        index = start_frame + round(span * percentile)
        if index not in indices:
            indices.append(index)
    return indices
