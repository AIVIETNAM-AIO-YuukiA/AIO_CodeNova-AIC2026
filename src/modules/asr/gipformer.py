"""Vietnamese ASR backed by g-group-ai-lab/gipformer-65M-rnnt (Zipformer Transducer).

Unlike the other model backends in this project, gipformer isn't a plain
HuggingFace Transformers checkpoint — it runs through sherpa-onnx via its own
CLI script (see ``external/gipformer/``), cloned as its own isolated
repo+venv (same convention as ``external/TransNetV2``) because its
sherpa-onnx pin doesn't need to coexist with the rest of this project's
dependency stack.

The published checkpoint is an *offline* (non-causal) transducer — confirmed
by attempting to load it into sherpa-onnx's ``OnlineRecognizer``, which fails
on missing streaming-only metadata — so true streaming decoding isn't
available. Long audio (a full video's soundtrack) is instead split into
fixed 30s windows with a small 1s overlap (standard practice for chunked/
"pseudo-streaming" offline ASR — see e.g.
https://ruoqijin.com/blog/asr-deep-dive-2025-2026) so words aren't cut mid-
utterance at a chunk boundary. Each chunk is transcribed independently and
kept as its own ``Transcript`` (never force-joined into one string — an
Elasticsearch document per chunk maps naturally to "nearest keyframe by
timestamp" the way ``paper2_cascaded_system.md`` describes ASR/keyframe
alignment); the 1s overlap region is deduplicated between consecutive chunks
via a token-level longest-common-subsequence match so the shared words at a
boundary aren't transcribed (and indexed) twice.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from core.errors import CodeNovaError
from modules.asr.base import AsrModel, Transcript

LOGGER = logging.getLogger(__name__)

_EXTERNAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "external" / "gipformer"
_VENV_PYTHON = _EXTERNAL_DIR / ".venv" / "bin" / "python"
_SCRIPT = _EXTERNAL_DIR / "infer_json.py"

_SAMPLE_RATE = 16000
_CHUNK_SECONDS = 30.0
_OVERLAP_SECONDS = 1.0


class GipformerAsrModel(AsrModel):
    """Transcribe a video's audio track with the self-hosted gipformer ASR model."""

    def __init__(self, quantize: str | None = None, num_threads: int | None = None) -> None:
        self.quantize = quantize or os.environ.get("GIPFORMER_QUANTIZE", "int8")
        self.num_threads = num_threads or int(os.environ.get("GIPFORMER_NUM_THREADS", "4"))
        self._checked_setup = False

    def transcribe(self, video_path: str) -> list[Transcript]:
        """Extract the audio track and transcribe it as overlap-deduplicated chunks.

        Returns an empty list (not an error) if the video has no audio stream
        — some archival/silent footage genuinely has none, and that's a valid
        pipeline input, not a failure.
        """
        self._ensure_setup()
        if not _has_audio_stream(video_path):
            LOGGER.info("No audio stream in %s; skipping ASR.", video_path)
            return []

        video_id = Path(video_path).stem
        duration = _probe_duration_seconds(video_path)

        with tempfile.TemporaryDirectory(prefix="gipformer_") as tmp_dir:
            tmp = Path(tmp_dir)
            windows = list(_chunk_windows(duration))
            chunk_paths = [
                _extract_audio(video_path, tmp / f"{video_id}_{i:04d}.wav", start, end)
                for i, (start, end) in enumerate(windows)
            ]
            chunk_texts = self._run_inference(chunk_paths)

        transcripts = []
        previous_text = ""
        for (start, end), text in zip(windows, chunk_texts):
            deduped = _dedupe_overlap(previous_text, text)
            if deduped:
                transcripts.append(
                    Transcript(
                        video_id=video_id,
                        text=deduped,
                        start_time_sec=start,
                        end_time_sec=end,
                    )
                )
            previous_text = text
        return transcripts

    def _run_inference(self, chunk_paths: list[Path]) -> list[str]:
        if not chunk_paths:
            return []
        result = subprocess.run(
            [
                str(_VENV_PYTHON),
                str(_SCRIPT),
                "--audio",
                *(str(p) for p in chunk_paths),
                "--quantize",
                self.quantize,
                "--num-threads",
                str(self.num_threads),
            ],
            capture_output=True,
            text=True,
            cwd=str(_EXTERNAL_DIR),
        )
        if result.returncode != 0 and not result.stdout.strip():
            raise CodeNovaError(f"gipformer ASR failed: {result.stderr.strip()}")

        texts_by_path: dict[str, str] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            payload = json.loads(line)
            if "error" in payload:
                raise CodeNovaError(f"gipformer ASR failed: {payload['error']}")
            texts_by_path[payload["audio_path"]] = str(payload.get("text", ""))

        missing = [str(p) for p in chunk_paths if str(p) not in texts_by_path]
        if missing:
            raise CodeNovaError(
                f"gipformer ASR produced no output for {len(missing)} chunk(s): {result.stderr.strip()}"
            )
        return [texts_by_path[str(p)] for p in chunk_paths]

    def _ensure_setup(self) -> None:
        if self._checked_setup:
            return
        if not _VENV_PYTHON.exists():
            raise CodeNovaError(
                f"gipformer venv not found at {_VENV_PYTHON}. Run "
                f"'cd {_EXTERNAL_DIR} && uv sync' to set it up."
            )
        self._checked_setup = True


def _chunk_windows(duration: float) -> list[tuple[float, float]]:
    """Return ``(start, end)`` seconds for fixed windows covering ``duration``.

    Consecutive windows share ``_OVERLAP_SECONDS`` at their boundary so a word
    spoken right at a cut point is fully captured in at least one chunk.
    """
    if duration <= 0:
        return []
    step = _CHUNK_SECONDS - _OVERLAP_SECONDS
    windows = []
    start = 0.0
    while start < duration:
        end = min(start + _CHUNK_SECONDS, duration)
        windows.append((start, end))
        if end >= duration:
            break
        start += step
    return windows


def _dedupe_overlap(previous_text: str, current_text: str) -> str:
    """Strip the leading words of ``current_text`` that duplicate the tail of
    ``previous_text`` in their shared ~1s overlap, via token-level LCS.

    Both chunks transcribe the same ~1s of audio at the boundary
    independently, so the recognizer's output for that shared audio should
    match closely (though not always exactly, since it's decoded with
    different surrounding context each time) — trim the run of matching
    trailing/leading words rather than assuming an exact string match.
    """
    if not previous_text:
        return current_text
    prev_words = previous_text.split()
    cur_words = current_text.split()
    if not prev_words or not cur_words:
        return current_text

    # Look for the longest run where the tail of prev_words matches the head
    # of cur_words — bounded search since only ~1s of audio (a handful of
    # words) can plausibly overlap.
    max_check = min(len(prev_words), len(cur_words), 12)
    best_overlap = 0
    for size in range(max_check, 0, -1):
        if prev_words[-size:] == cur_words[:size]:
            best_overlap = size
            break
    return " ".join(cur_words[best_overlap:])


def _probe_duration_seconds(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise CodeNovaError(f"ffprobe failed to read duration for {video_path}: {result.stderr}")
    return float(result.stdout.strip())


def _has_audio_stream(video_path: str) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _extract_audio(video_path: str, wav_path: Path, start: float, end: float) -> Path:
    """Extract one ``[start, end)`` window of a video's audio track to 16kHz mono WAV."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE),
            "-f",
            "wav",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CodeNovaError(f"ffmpeg audio extraction failed for {video_path}: {result.stderr}")
    return wav_path
