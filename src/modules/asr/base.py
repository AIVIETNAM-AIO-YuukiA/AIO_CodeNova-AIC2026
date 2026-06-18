"""ASR interface.

Stub only: defines the contract so a Whisper/Qwen3-ASR backend can be added
later without touching the indexing or retrieval layers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transcript:
    """Speech transcript for one video segment."""

    video_id: str
    text: str
    start_time_sec: float | None = None
    end_time_sec: float | None = None


class AsrModel:
    """Interface for speech-to-text backends (Whisper, WhisperX, Qwen3-ASR)."""

    def transcribe(self, video_path: str) -> list[Transcript]:
        """Transcribe a video's audio track into time-stamped segments."""
        raise NotImplementedError
