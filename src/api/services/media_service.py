"""Media domain service — frame file resolution and per-video shot listing.

``resolve_frame_file`` keeps the same safety check ui/server.py's
``_send_frame`` had: only known image extensions are resolved, and the path
must land inside the experiment's own frame tree (see
``core.paths.resolve_experiment_frame_path``) before any bytes are read.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from config.settings import Experiment
from ui.api import handle_video_shots

_ALLOWED_FRAME_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def resolve_frame_file(experiment: Experiment, raw_path: str) -> Path | None:
    from core.paths import resolve_experiment_frame_path

    if Path(raw_path).suffix.lower() not in _ALLOWED_FRAME_SUFFIXES:
        return None
    resolution = resolve_experiment_frame_path(experiment, raw_path)
    if not resolution.valid or resolution.resolved_path is None:
        return None
    return resolution.resolved_path


def get_video_shots(experiment: Experiment, video_id: str) -> tuple[dict, HTTPStatus]:
    return handle_video_shots({"video_id": [video_id]}, experiment)
