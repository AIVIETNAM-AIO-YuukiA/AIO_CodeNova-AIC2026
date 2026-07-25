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
available. Long audio is cut at speech boundaries using Silero-VAD
(``modules/asr/vad.py``) rather than at fixed-time marks, then each speech
segment is transcribed independently and the resulting texts are regrouped
into semantic chunks of ~``target_words`` words (see ``_group_by_words``) —
same two-stage approach (VAD segmentation + word-count regrouping) as the
AIC 2025 reference project's ASR pipeline, adapted to gipformer instead of
Whisper. A chunk boundary never lands mid-sentence because it only ever
falls on a VAD-detected silence. Each chunk becomes its own ``Transcript``
(never force-joined into one whole-video string) so an Elasticsearch
document per chunk maps naturally to "nearest keyframe by timestamp", per
``paper2_cascaded_system.md``'s ASR/keyframe alignment.

All of a video's segments are transcribed via one (or a few, if there are
100+) calls into ``external/gipformer/infer_json.py``, which decodes them
with sherpa_onnx's ``decode_streams()`` — a real batched decode processed
concurrently inside the engine, matching how the AIC 2025 reference
project's Whisper pipeline batches its HF ``pipeline(..., batch_size=...)``
call. An earlier version of this code called ``infer_json.py`` with the
same batching but that script decoded one file at a time internally
(looping ``decode_streams([1 stream])``); that's the actual reason a
135-segment video OOM-killed its subprocess, not the batch size — see
``infer_json.py``'s docstring for the fix.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from core.errors import CodeNovaError
from modules.asr.base import AsrModel, Transcript
from modules.asr.vad import SileroVad

LOGGER = logging.getLogger(__name__)

_EXTERNAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "external" / "gipformer"
_VENV_PYTHON = _EXTERNAL_DIR / ".venv" / "bin" / "python"
_SCRIPT = _EXTERNAL_DIR / "infer_json.py"

_SAMPLE_RATE = 16000

# Fallback fixed-window size when VAD is unavailable, and the cap on how long
# a single VAD speech segment may run before being force-split — gipformer is
# non-causal, so an unbounded segment (e.g. 2 minutes of continuous speech)
# would blow up decode time/memory.
_MAX_SEGMENT_SECONDS = 30.0
_DEFAULT_TARGET_WORDS = 150

# infer_json.py decodes all its --audio segments in one batched
# sherpa_onnx.decode_streams() call, which loads every segment's audio into
# memory before decoding — capping the batch bounds peak memory for videos
# with unusually many VAD segments (100+ for a long/chatty news clip) without
# giving up the real speedup decode_streams() gives over decoding one file
# at a time (see infer_json.py's docstring).
_MAX_SEGMENTS_PER_INFERENCE_CALL = 100


class GipformerAsrModel(AsrModel):
    """Transcribe a video's audio track with the self-hosted gipformer ASR model."""

    def __init__(
        self,
        quantize: str | None = None,
        num_threads: int | None = None,
        target_words: int | None = None,
    ) -> None:
        self.quantize = quantize or os.environ.get("GIPFORMER_QUANTIZE", "int8")
        self.num_threads = num_threads or int(os.environ.get("GIPFORMER_NUM_THREADS", "4"))
        self.target_words = target_words or int(
            os.environ.get("ASR_TARGET_WORDS", _DEFAULT_TARGET_WORDS)
        )
        self._checked_setup = False
        self._vad = SileroVad()

    def transcribe(self, video_path: str) -> list[Transcript]:
        """Extract the audio track, VAD-segment it, transcribe, then regroup by word count.

        Returns an empty list (not an error) if the video has no audio stream
        — some archival/silent footage genuinely has none, and that's a valid
        pipeline input, not a failure.
        """
        self._ensure_setup()
        if not _has_audio_stream(video_path):
            LOGGER.info("No audio stream in %s; skipping ASR.", video_path)
            return []

        video_id = Path(video_path).stem

        with tempfile.TemporaryDirectory(prefix="gipformer_") as tmp_dir:
            tmp = Path(tmp_dir)
            full_wav = _extract_audio(video_path, tmp / f"{video_id}_full.wav")
            windows = _speech_windows(self._vad, str(full_wav))
            # Slice segments out of the already-extracted WAV in-process
            # (wave module, no subprocess) rather than calling ffmpeg once per
            # segment — a VAD-heavy video can have 50-100+ speech segments,
            # and that many ffmpeg spawns dominates total processing time.
            segment_paths = [
                _slice_wav(full_wav, tmp / f"{video_id}_{i:04d}.wav", start, end)
                for i, (start, end) in enumerate(windows)
            ]
            segment_texts = self._run_inference(segment_paths)

        segments = [
            {"start": start, "end": end, "text": text}
            for (start, end), text in zip(windows, segment_texts)
            if text.strip()
        ]
        chunks = _group_by_words(segments, self.target_words)
        return [
            Transcript(
                video_id=video_id,
                text=chunk["text"],
                start_time_sec=chunk["start"],
                end_time_sec=chunk["end"],
            )
            for chunk in chunks
        ]

    def _run_inference(self, segment_paths: list[Path]) -> list[str]:
        """Transcribe all segments, batching subprocess calls so a video with
        many segments (VAD can produce 100+ for a long news clip) can't OOM
        a single long-lived infer_json.py process (see
        ``_MAX_SEGMENTS_PER_INFERENCE_CALL``)."""
        if not segment_paths:
            return []
        texts_by_path: dict[str, str] = {}
        for start in range(0, len(segment_paths), _MAX_SEGMENTS_PER_INFERENCE_CALL):
            batch = segment_paths[start : start + _MAX_SEGMENTS_PER_INFERENCE_CALL]
            texts_by_path.update(self._call_subprocess(batch))
        return [texts_by_path[str(p)] for p in segment_paths]

    def _call_subprocess(self, segment_paths: list[Path]) -> dict[str, str]:
        result = subprocess.run(
            [
                str(_VENV_PYTHON),
                str(_SCRIPT),
                "--audio",
                *(str(p) for p in segment_paths),
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
            # A negative returncode means the subprocess was killed by a
            # signal (e.g. SIGKILL from the OOM killer) rather than exiting
            # normally — stderr is empty in that case, so surface the signal
            # number explicitly instead of raising an unhelpful blank message.
            if result.returncode < 0:
                raise CodeNovaError(
                    f"gipformer ASR killed by signal {-result.returncode} "
                    f"(likely OOM) while transcribing {len(segment_paths)} segments"
                )
            raise CodeNovaError(
                f"gipformer ASR failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        texts_by_path: dict[str, str] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            payload = json.loads(line)
            if "error" in payload:
                raise CodeNovaError(f"gipformer ASR failed: {payload['error']}")
            texts_by_path[payload["audio_path"]] = str(payload.get("text", ""))

        missing = [str(p) for p in segment_paths if str(p) not in texts_by_path]
        if missing:
            raise CodeNovaError(
                f"gipformer ASR produced no output for {len(missing)} chunk(s): {result.stderr.strip()}"
            )
        return texts_by_path

    def _ensure_setup(self) -> None:
        if self._checked_setup:
            return
        if not _VENV_PYTHON.exists():
            raise CodeNovaError(
                f"gipformer venv not found at {_VENV_PYTHON}. Run "
                f"'cd {_EXTERNAL_DIR} && uv sync' to set it up."
            )
        self._checked_setup = True


def _speech_windows(vad: SileroVad, wav_path: str) -> list[tuple[float, float]]:
    """Return ``(start, end)`` seconds to transcribe: VAD speech regions, each
    capped at ``_MAX_SEGMENT_SECONDS`` (gipformer is non-causal — an unbounded
    segment would blow up decode time/memory). Falls back to one segment
    covering the whole file if VAD found nothing (silent audio) or is
    unavailable (see ``SileroVad.speech_segments``).
    """
    segments = vad.speech_segments(wav_path)
    if not segments:
        duration = _probe_duration_seconds(wav_path)
        segments = [(0.0, duration)] if duration > 0 else []

    windows = []
    for start, end in segments:
        while end - start > _MAX_SEGMENT_SECONDS:
            windows.append((start, start + _MAX_SEGMENT_SECONDS))
            start += _MAX_SEGMENT_SECONDS
        if end > start:
            windows.append((start, end))
    return windows


def _group_by_words(segments: list[dict], target_words: int) -> list[dict]:
    """Regroup VAD-transcribed segments into chunks of ~``target_words`` words.

    Mirrors the AIC 2025 reference pipeline's semantic chunking, minus the
    embedding-similarity split (no spare embedding model in gipformer's
    lightweight stack) — grouping by word count already keeps chunks a
    sensible size for an Elasticsearch document, and boundaries always fall
    on a VAD silence so sentences are never split mid-way.
    """
    if not segments:
        return []
    chunks = []
    current: list[dict] = []
    word_count = 0
    for seg in segments:
        words = len(seg["text"].split())
        if current and word_count + words > target_words:
            chunks.append(_finalize_word_group(current))
            current, word_count = [], 0
        current.append(seg)
        word_count += words
    if current:
        chunks.append(_finalize_word_group(current))
    return chunks


def _finalize_word_group(segments: list[dict]) -> dict:
    return {
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(seg["text"].strip() for seg in segments),
    }


def _probe_duration_seconds(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise CodeNovaError(f"ffprobe failed to read duration for {path}: {result.stderr}")
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


def _extract_audio(video_path: str, wav_path: Path) -> Path:
    """Extract the video's full audio track to 16kHz mono WAV."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
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


def _slice_wav(source_wav: Path, out_path: Path, start: float, end: float) -> Path:
    """Cut ``[start, end)`` seconds out of ``source_wav`` without spawning ffmpeg.

    A pure-Python frame-offset slice (stdlib ``wave``) instead of a
    per-segment ffmpeg subprocess — VAD can produce 50-100+ speech segments
    per video, and that many process spawns was the actual bottleneck
    (each ffmpeg invocation has ~fixed startup overhead regardless of how
    little audio it's cutting).
    """
    with wave.open(str(source_wav), "rb") as src:
        params = src.getparams()
        frame_rate = params.framerate
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        src.setpos(max(start_frame, 0))
        frames = src.readframes(end_frame - start_frame)

    with wave.open(str(out_path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
    return out_path
