"""Shot detection boundary interfaces."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys

from codenova.core.errors import ShotDetectionError
from codenova.core.types import ShotRecord, VideoRecord


class ShotDetector:
    """Interface for shot boundary detection backends."""

    def detect(self, video: VideoRecord) -> list[ShotRecord]:
        """Detect shots for a video."""
        raise NotImplementedError


class TransNetV2ShotDetector(ShotDetector):
    """TransNetV2-backed shot detector using the PyTorch inference implementation."""

    def __init__(
        self,
        weights_path: Path | None = None,
        module_dir: Path | None = None,
        threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.weights_path = weights_path
        self.module_dir = module_dir
        self.threshold = threshold
        self.device = device
        self._model = None
        self._torch = None

    def detect(self, video: VideoRecord) -> list[ShotRecord]:
        """Detect shots using TransNetV2 cut probabilities."""
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ShotDetectionError("Install OpenCV and NumPy before running TransNetV2.") from exc

        model, torch, device = self._load_model()
        capture = cv2.VideoCapture(video.path)
        if not capture.isOpened():
            raise ShotDetectionError(f"Cannot open video: {video.path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise ShotDetectionError(f"No frames decoded from video: {video.path}")

        array = np.asarray(frames, dtype=np.uint8)
        scores = predict_transnetv2_scores(model=model, torch=torch, frames=array, device=device)

        cut_frames = [index for index, score in enumerate(scores) if float(score) >= self.threshold]
        return predictions_to_shots(video.video_id, cut_frames, len(frames), fps)

    def _load_model(self):
        if self._model is not None and self._torch is not None:
            return self._model

        if self.weights_path is None or not self.weights_path.exists():
            raise ShotDetectionError(
                "TransNetV2 requires converted PyTorch weights. Pass "
                "--transnetv2-weights /path/to/transnetv2-pytorch-weights.pth."
            )
        if self.module_dir is not None:
            sys.path.insert(0, str(self.module_dir))

        try:
            torch = importlib.import_module("torch")
            module = importlib.import_module("transnetv2_pytorch")
        except ImportError as exc:
            raise ShotDetectionError(
                "TransNetV2 PyTorch module was not found. Pass --transnetv2-module-dir "
                "pointing at the TransNetV2 inference-pytorch folder."
            ) from exc

        device = _resolve_torch_device(torch, self.device)
        model = module.TransNetV2()
        state_dict = torch.load(self.weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        self._model = (model, torch, device)
        self._torch = torch
        return self._model


def predictions_to_shots(
    video_id: str,
    cut_frames: list[int],
    frame_count: int,
    fps: float,
) -> list[ShotRecord]:
    """Convert predicted cut frame indices to contiguous shot records."""
    boundaries = sorted({frame for frame in cut_frames if 0 <= frame < frame_count - 1})
    starts = [0, *(frame + 1 for frame in boundaries)]
    ends = [*boundaries, frame_count - 1]
    shots = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        shots.append(
            ShotRecord(
                video_id=video_id,
                shot_id=f"{video_id}_s{index:06d}",
                start_frame=start,
                end_frame=end,
                start_time_sec=(start / fps) if fps else None,
                end_time_sec=(end / fps) if fps else None,
            )
        )
    return shots


def predict_transnetv2_scores(model, torch, frames, device) -> list[float]:
    """Run TransNetV2 in 100-frame windows and return one score per frame.

    This mirrors the official TransNetV2 inference wrapper: each model call sees
    100 frames, only predictions for frames 25..74 are kept, and the window
    advances by 50 frames. This avoids allocating the whole video on the GPU.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ShotDetectionError("Install NumPy before running TransNetV2.") from exc

    if len(frames) == 0:
        return []

    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for window in transnetv2_windows(frames):
            tensor = torch.from_numpy(window).to(device)
            single_frame_pred, _ = model(tensor)
            scores = torch.sigmoid(single_frame_pred)[0, 25:75, 0]
            predictions.append(scores.detach().cpu().numpy())
            del tensor, single_frame_pred, scores

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(predictions).astype("float32")[: len(frames)].tolist()


def transnetv2_windows(frames):
    """Yield padded TransNetV2 input windows with shape ``[1, 100, 27, 48, 3]``."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ShotDetectionError("Install NumPy before running TransNetV2.") from exc

    no_padded_frames_start = 25
    remainder = len(frames) % 50
    no_padded_frames_end = 25 + 50 - (remainder if remainder != 0 else 50)

    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded_inputs = np.concatenate(
        [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end,
        axis=0,
    )

    ptr = 0
    while ptr + 100 <= len(padded_inputs):
        yield padded_inputs[ptr : ptr + 100][np.newaxis]
        ptr += 50


def _resolve_torch_device(torch, requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise ShotDetectionError(
            "CUDA was requested by default but torch.cuda.is_available() is false."
        )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ShotDetectionError(f"Requested device '{requested}' but CUDA is not available.")
    return torch.device(requested)
