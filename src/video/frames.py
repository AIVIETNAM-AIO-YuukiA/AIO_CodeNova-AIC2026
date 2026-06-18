"""Keyframe extraction interfaces."""

from __future__ import annotations

from pathlib import Path

from core.errors import FrameExtractionError
from core.types import FrameRecord, ShotRecord, VideoRecord


class FrameExtractor:
    """Interface for extracting representative frames from shots."""

    def extract(self, video: VideoRecord, shots: list[ShotRecord]) -> list[FrameRecord]:
        """Extract keyframes for a video's shots."""
        raise NotImplementedError


DEFAULT_KEYFRAME_PERCENTILES = (0.15, 0.5, 0.85)


class OpenCVFrameExtractor(FrameExtractor):
    """OpenCV-backed keyframe extractor sampling shots at fixed percentiles."""

    def __init__(
        self,
        output_dir: Path,
        keyframe_percentiles: tuple[float, ...] = DEFAULT_KEYFRAME_PERCENTILES,
    ) -> None:
        self.output_dir = output_dir
        self.keyframe_percentiles = tuple(keyframe_percentiles)

    def extract(self, video: VideoRecord, shots: list[ShotRecord]) -> list[FrameRecord]:
        """Extract percentile keyframes for each shot."""
        try:
            import cv2
        except ImportError as exc:
            raise FrameExtractionError("Install opencv-python before extracting frames.") from exc

        capture = cv2.VideoCapture(video.path)
        if not capture.isOpened():
            raise FrameExtractionError(f"Cannot open video: {video.path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        records: list[FrameRecord] = []
        video_dir = self.output_dir / video.video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        try:
            for shot in shots:
                for frame_index in sample_frame_indices(
                    shot.start_frame, shot.end_frame, self.keyframe_percentiles
                ):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        raise FrameExtractionError(
                            f"Cannot decode frame={frame_index} video={video.path}"
                        )
                    frame_id = f"{shot.shot_id}_f{frame_index:08d}"
                    frame_path = video_dir / f"{frame_id}.jpg"
                    if not cv2.imwrite(str(frame_path), frame):
                        raise FrameExtractionError(f"Cannot write frame image: {frame_path}")
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
        finally:
            capture.release()
        return records


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
