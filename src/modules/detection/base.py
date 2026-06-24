"""Object detection interface.

Stub only: defines the contract so a YOLO11 backend can be added later
without touching the indexing or retrieval layers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected object on a frame."""

    frame_id: str
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


class DetectionModel:
    """Interface for object detection backends (YOLO11)."""

    def detect(self, frame_path: str) -> list[Detection]:
        """Return detected objects for one frame image."""
        raise NotImplementedError
