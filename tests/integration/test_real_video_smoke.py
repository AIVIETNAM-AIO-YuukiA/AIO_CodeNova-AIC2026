"""Opt-in smoke test for the real ffmpeg/OpenCV frame path.

Run with::

    RUN_REAL_VIDEO_TESTS=1 pytest -q tests/integration/test_real_video_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from core.types import ShotRecord, VideoRecord
from video.frames import FFmpegFrameExtractor, probe_fps


pytestmark = [pytest.mark.integration, pytest.mark.requires_ffmpeg]


def _real_video_enabled() -> bool:
    return os.getenv("RUN_REAL_VIDEO_TESTS") == "1"


@pytest.mark.skipif(not _real_video_enabled(), reason="set RUN_REAL_VIDEO_TESTS=1")
def test_real_video_can_be_probed_and_frames_extracted(tmp_path: Path) -> None:
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("cv2")
    video_path = tmp_path / "sample.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=10:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    video = VideoRecord(
        video_id="sample",
        path=str(video_path),
        checksum="checksum",
        size_bytes=video_path.stat().st_size,
    )
    shot = ShotRecord(
        shot_id="sample_s000000",
        video_id=video.video_id,
        start_frame=0,
        end_frame=9,
        start_time_sec=0.0,
        end_time_sec=0.9,
    )
    extractor = FFmpegFrameExtractor(
        output_dir=tmp_path / "frames",
        keyframe_percentiles=(0.0, 0.5, 1.0),
    )

    assert probe_fps(str(video_path)) == pytest.approx(10.0, rel=0.1)
    frames = extractor.extract(video, [shot])

    assert [frame.frame_index for frame in frames] == [0, 4, 9]
    assert all(Path(frame.frame_path).is_file() for frame in frames)
    assert [frame.timestamp_sec for frame in frames] == pytest.approx([0.0, 0.4, 0.9])
