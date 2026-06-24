"""Build ASR backends from environment configuration."""

from __future__ import annotations

import os

from core.errors import CodeNovaError
from modules.asr.base import AsrModel
from modules.asr.qwen3_gguf import Qwen3GgufAsrModel


def build_asr_model() -> AsrModel:
    """Create the configured ASR backend."""
    backend = os.environ.get("ASR_BACKEND", "qwen3_gguf")
    if backend == "qwen3_gguf":
        return Qwen3GgufAsrModel(
            model_path=os.environ.get("ASR_QWEN_MODEL_PATH", ""),
            crispasr_bin=os.environ.get("ASR_CRISPASR_BIN", ""),
            language=os.environ.get("ASR_LANGUAGE", "auto"),
            sample_rate=int(os.environ.get("ASR_AUDIO_SAMPLE_RATE", "16000")),
        )
    raise CodeNovaError(f"Unsupported ASR_BACKEND '{backend}'. Only 'qwen3_gguf' is supported.")
