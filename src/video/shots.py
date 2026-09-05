"""Shot detection boundary interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import os
import subprocess
import sys

from core.errors import ShotDetectionError
from core.types import ShotRecord, VideoRecord

# TransNetV2's fixed input resolution.
_INPUT_WIDTH = 48
_INPUT_HEIGHT = 27

# Windows per model call. Measured on a 4GB RTX 3050: batch 1 runs 28.7s
# (169MB) vs 33.2s (1GB) at batch 8 — the 3D-CNN's activations are bandwidth-
# bound, so bigger batches only pay off on GPUs with more memory bandwidth.
_BATCH_SIZE = int(os.environ.get("TRANSNETV2_BATCH_SIZE", "1"))


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
        return self.detect_decoded(video, decode_video(video.path))

    def detect_decoded(self, video: VideoRecord, decoded: DecodedVideo) -> list[ShotRecord]:
        """Run detection on already-decoded frames (see ``decode_video``)."""
        model, torch, device = self._load_model()
        scores = predict_transnetv2_scores(
            model=model, torch=torch, frames=decoded.frames, device=device
        )
        cut_frames = [index for index, score in enumerate(scores) if float(score) >= self.threshold]
        return predictions_to_shots(video.video_id, cut_frames, len(decoded.frames), decoded.fps)

    def _load_model(self):
        if self._model is not None and self._torch is not None:
            return self._model

        if self.weights_path is None or not self.weights_path.exists():
            from core.external_setup import ensure_transnetv2

            self.weights_path = ensure_transnetv2(self.weights_path)
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


@dataclass(frozen=True)
class DecodedVideo:
    """A video decoded to TransNetV2's input resolution, plus its frame rate."""

    frames: object  # numpy uint8 array, shape (N, 27, 48, 3)
    fps: float


def decode_video(video_path: str) -> DecodedVideo:
    """Decode a video for TransNetV2. Safe to call off the main thread."""
    return DecodedVideo(frames=_decode_frames(video_path), fps=_probe_fps(video_path))


def _ffmpeg_exe() -> str:
    """Return the bundled ffmpeg binary path."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ShotDetectionError("Install imageio-ffmpeg before running TransNetV2.") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _probe_fps(video_path: str) -> float:
    """Read a video's frame rate with OpenCV (metadata only, no decoding)."""
    try:
        import cv2
    except ImportError as exc:
        raise ShotDetectionError("Install OpenCV before running TransNetV2.") from exc

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ShotDetectionError(f"Cannot open video: {video_path}")
    try:
        return capture.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        capture.release()


def _decode_frames(video_path: str):
    """Decode a whole video to a uint8 (N, 27, 48, 3) RGB array via ffmpeg.

    ffmpeg decodes and scales in one multi-threaded pass, which is ~9x faster
    than reading full-resolution frames through OpenCV and resizing each one.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ShotDetectionError("Install NumPy before running TransNetV2.") from exc

    command = [
        _ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        f"scale={_INPUT_WIDTH}:{_INPUT_HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    process = subprocess.run(command, capture_output=True)
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", "replace").strip()[:500]
        raise ShotDetectionError(f"ffmpeg failed to decode {video_path}: {message}")

    frame_bytes = _INPUT_WIDTH * _INPUT_HEIGHT * 3
    count = len(process.stdout) // frame_bytes
    if count == 0:
        raise ShotDetectionError(f"No frames decoded from video: {video_path}")
    usable = memoryview(process.stdout)[: count * frame_bytes]
    return np.frombuffer(usable, dtype=np.uint8).reshape(count, _INPUT_HEIGHT, _INPUT_WIDTH, 3)


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


def predict_transnetv2_scores(model, torch, frames, device, batch_size: int | None = None):
    """Run TransNetV2 in 100-frame windows and return one score per frame.

    This mirrors the official TransNetV2 inference wrapper: each model call sees
    100 frames, only predictions for frames 25..74 are kept, and the window
    advances by 50 frames. Windows are submitted ``batch_size`` at a time
    (``TRANSNETV2_BATCH_SIZE``) so the whole video never sits on the GPU.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ShotDetectionError("Install NumPy before running TransNetV2.") from exc

    if len(frames) == 0:
        return []

    size = batch_size or _BATCH_SIZE
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in _batched(transnetv2_windows(frames), size):
            tensor = torch.from_numpy(np.concatenate(batch, axis=0)).to(device)
            single_frame_pred, _ = model(tensor)
            scores = torch.sigmoid(single_frame_pred)[:, 25:75, 0]
            predictions.append(scores.detach().cpu().numpy().reshape(-1))
            del tensor, single_frame_pred, scores

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(predictions).astype("float32")[: len(frames)].tolist()


def _batched(iterable, size: int):
    """Yield lists of up to ``size`` items from ``iterable``."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


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
