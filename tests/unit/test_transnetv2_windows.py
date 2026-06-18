import numpy as np

from codenova.video.shots import transnetv2_windows


def test_transnetv2_windows_use_fixed_100_frame_inputs() -> None:
    frames = np.zeros((126, 27, 48, 3), dtype=np.uint8)

    windows = list(transnetv2_windows(frames))

    assert len(windows) == 3
    assert all(window.shape == (1, 100, 27, 48, 3) for window in windows)
