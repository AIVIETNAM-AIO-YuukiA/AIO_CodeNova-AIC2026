import pytest

from codenova.core.errors import FrameExtractionError
from codenova.video.frames import DEFAULT_KEYFRAME_PERCENTILES, sample_frame_indices


def test_default_percentiles() -> None:
    assert DEFAULT_KEYFRAME_PERCENTILES == (0.15, 0.5, 0.85)


def test_samples_at_each_percentile() -> None:
    # span = 1000 frames over [0, 1000]
    assert sample_frame_indices(0, 1000, (0.15, 0.5, 0.85)) == [150, 500, 850]


def test_offset_start_frame_is_respected() -> None:
    # start=100, span=200 -> 100 + round(200*p)
    assert sample_frame_indices(100, 300, (0.15, 0.5, 0.85)) == [130, 200, 270]


def test_short_shot_collapses_duplicate_indices() -> None:
    # zero-length shot yields a single keyframe, not three identical ones
    assert sample_frame_indices(42, 42, (0.15, 0.5, 0.85)) == [42]


def test_empty_percentiles_rejected() -> None:
    with pytest.raises(FrameExtractionError, match="must not be empty"):
        sample_frame_indices(0, 100, ())


def test_percentile_out_of_range_rejected() -> None:
    with pytest.raises(FrameExtractionError, match="outside"):
        sample_frame_indices(0, 100, (1.5,))


def test_end_before_start_rejected() -> None:
    with pytest.raises(FrameExtractionError, match="smaller than start"):
        sample_frame_indices(100, 50, (0.5,))
