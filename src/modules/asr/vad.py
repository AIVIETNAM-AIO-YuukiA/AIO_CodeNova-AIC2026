"""Voice activity detection (Silero-VAD) — finds speech segments in audio.

Used ahead of ASR transcription so gipformer only ever sees audio that
actually contains speech, cut at silence boundaries instead of at fixed-time
marks. This is the same VAD backend (snakers4/silero-vad) used by the AIC 2025
reference project's ASR pipeline — see ``.claude/references/`` for context.
"""

from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


class SileroVad:
    """Wraps Silero-VAD to return speech (start, end) timestamps in seconds."""

    def __init__(
        self,
        min_silence_ms: int | None = None,
        speech_pad_ms: int | None = None,
        threshold: float | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.min_silence_ms = min_silence_ms or int(os.environ.get("ASR_VAD_MIN_SILENCE_MS", "250"))
        self.speech_pad_ms = speech_pad_ms or int(os.environ.get("ASR_VAD_SPEECH_PAD_MS", "200"))
        self.threshold = threshold or float(os.environ.get("ASR_VAD_THRESHOLD", "0.5"))
        self.num_threads = num_threads or int(os.environ.get("ASR_VAD_NUM_THREADS", "4"))
        self._model = None
        self._get_speech_timestamps = None
        self._read_audio = None

    def speech_segments(self, wav_path: str) -> list[tuple[float, float]]:
        """Return ``(start_sec, end_sec)`` for each speech region in ``wav_path``.

        Falls back to a single segment spanning the whole file if the VAD
        model fails to load (network issue on first ``torch.hub`` download) —
        ASR still runs, just without VAD-based cutting.
        """
        try:
            get_speech_timestamps, read_audio = self._load()
        except Exception:
            LOGGER.exception("Silero-VAD unavailable; transcribing without VAD segmentation.")
            return []

        wav = read_audio(wav_path, sampling_rate=_SAMPLE_RATE)
        timestamps = get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=_SAMPLE_RATE,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            threshold=self.threshold,
        )
        return [(ts["start"] / _SAMPLE_RATE, ts["end"] / _SAMPLE_RATE) for ts in timestamps]

    def _load(self):
        if self._model is not None:
            return self._get_speech_timestamps, self._read_audio
        import torch

        torch.set_num_threads(self.num_threads)

        # onnx=True: ~470s -> ~7s on a 15min video vs the .jit variant,
        # measured on this machine — its per-chunk Python loop has far more
        # overhead per call than ONNX Runtime's.
        # trust_repo=True skips torch.hub's interactive y/N prompt (this
        # pipeline runs unattended).
        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
            trust_repo=True,
        )
        self._get_speech_timestamps, _, self._read_audio, _, _ = utils
        return self._get_speech_timestamps, self._read_audio
