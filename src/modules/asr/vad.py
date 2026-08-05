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


def _read_wav_mono(wav_path: str):
    """Read a 16 kHz mono WAV into a float32 tensor in [-1, 1].

    Silero's bundled ``read_audio`` goes through ``torchaudio``, whose
    ``list_audio_backends`` was removed in torchaudio 2.11. The pipeline always
    hands this a 16 kHz mono PCM file it just wrote with ffmpeg, so reading it
    with the stdlib avoids that dependency entirely.
    """
    import wave

    import numpy as np
    import torch

    with wave.open(wav_path, "rb") as handle:
        if handle.getframerate() != _SAMPLE_RATE:
            raise ValueError(f"expected {_SAMPLE_RATE} Hz, got {handle.getframerate()}")
        if handle.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {handle.getsampwidth() * 8}-bit")
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(samples)


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
            get_speech_timestamps, _ = self._load()
            wav = _read_wav_mono(wav_path)
        except Exception:
            LOGGER.exception("Silero-VAD unavailable; transcribing without VAD segmentation.")
            return []

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
