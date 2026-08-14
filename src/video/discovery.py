"""Video file discovery."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from core.types import VideoRecord

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def file_checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 checksum for a file."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_videos(input_dir: Path) -> list[VideoRecord]:
    """Find videos recursively and return stable video records."""
    videos: list[VideoRecord] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        checksum = file_checksum(path)
        video_id = checksum[:16]
        videos.append(
            VideoRecord(
                video_id=video_id,
                path=str(path),
                checksum=checksum,
                size_bytes=path.stat().st_size,
            )
        )
    return videos


def discover_video_paths(input_dir: Path):
    """Yield supported video paths without reading their contents."""
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def inspect_video(path: Path) -> VideoRecord:
    """Build a stable record for one path so callers can isolate per-file errors."""
    checksum = file_checksum(path)
    return VideoRecord(
        video_id=checksum[:16],
        path=str(path),
        checksum=checksum,
        size_bytes=path.stat().st_size,
    )
