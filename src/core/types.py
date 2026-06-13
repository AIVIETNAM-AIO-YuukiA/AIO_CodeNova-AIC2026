"""Typed records shared across pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VideoRecord:
    """Metadata for a discovered video."""

    video_id: str
    path: str
    checksum: str
    size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        """Serialize the record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "VideoRecord":
        """Deserialize a video record from manifest data."""
        return cls(
            video_id=str(payload["video_id"]),
            path=str(payload["path"]),
            checksum=str(payload["checksum"]),
            size_bytes=int(payload["size_bytes"]),
        )


@dataclass(frozen=True)
class ShotRecord:
    """Shot boundary metadata for one video segment."""

    video_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    start_time_sec: float | None = None
    end_time_sec: float | None = None

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Serialize the record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ShotRecord":
        """Deserialize a shot record from manifest data."""
        return cls(
            video_id=str(payload["video_id"]),
            shot_id=str(payload["shot_id"]),
            start_frame=int(payload["start_frame"]),
            end_frame=int(payload["end_frame"]),
            start_time_sec=_optional_float(payload.get("start_time_sec")),
            end_time_sec=_optional_float(payload.get("end_time_sec")),
        )


@dataclass(frozen=True)
class FrameRecord:
    """Metadata for one extracted keyframe."""

    frame_id: str
    video_id: str
    shot_id: str
    frame_path: str
    frame_index: int | None = None
    timestamp_sec: float | None = None

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Serialize the record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FrameRecord":
        """Deserialize a frame record from manifest data."""
        return cls(
            frame_id=str(payload["frame_id"]),
            video_id=str(payload["video_id"]),
            shot_id=str(payload["shot_id"]),
            frame_path=str(payload["frame_path"]),
            frame_index=_optional_int(payload.get("frame_index")),
            timestamp_sec=_optional_float(payload.get("timestamp_sec")),
        )


@dataclass(frozen=True)
class SearchResult:
    """One retrieval result."""

    frame_id: str
    video_id: str
    score: float
    frame_path: str | None = None
    video_path: str | None = None
    video_name: str | None = None
    shot_id: str | None = None
    frame_index: int | None = None
    timestamp_sec: float | None = None

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Serialize the result."""
        return asdict(self)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
