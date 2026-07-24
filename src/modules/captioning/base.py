"""Captioning interface.

See ``modules/captioning/vllm.py`` for the concrete backend (self-hosted
vLLM, OpenAI-compatible chat-completions API).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Caption:
    """Natural-language description of a frame."""

    frame_id: str
    video_id: str
    text: str


class CaptioningModel:
    """Interface for image captioning backends (InternVL3, Qwen2.5-VL)."""

    def caption(self, frame_path: str) -> str:
        """Return a natural-language caption for one frame image."""
        raise NotImplementedError
