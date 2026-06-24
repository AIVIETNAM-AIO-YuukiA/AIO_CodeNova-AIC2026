"""Qwen3-ASR GGUF wrapper using a CrispASR executable."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile

from core.errors import CodeNovaError
from modules.asr.base import AsrModel, Transcript


class Qwen3GgufAsrModel(AsrModel):
    """ASR backend for Qwen3-ASR GGUF through CrispASR."""

    def __init__(
        self,
        model_path: str,
        crispasr_bin: str,
        language: str = "auto",
        sample_rate: int = 16000,
    ) -> None:
        if not model_path:
            raise CodeNovaError("ASR_QWEN_MODEL_PATH is required for ASR_BACKEND=qwen3_gguf.")
        if not crispasr_bin:
            raise CodeNovaError("ASR_CRISPASR_BIN is required for ASR_BACKEND=qwen3_gguf.")
        self.model_path = model_path
        self.crispasr_bin = crispasr_bin
        self.language = language
        self.sample_rate = sample_rate

    def transcribe(self, video_path: str) -> list[Transcript]:
        """Transcribe a video's audio track into time-stamped segments."""
        source = Path(video_path)
        if not source.exists():
            raise CodeNovaError(f"ASR video does not exist: {video_path}")
        if not Path(self.model_path).exists():
            raise CodeNovaError(f"ASR model file does not exist: {self.model_path}")
        if not Path(self.crispasr_bin).exists():
            raise CodeNovaError(f"CrispASR binary does not exist: {self.crispasr_bin}")

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            extract_audio(video_path, audio_path, self.sample_rate)
            result = run_crispasr(
                crispasr_bin=self.crispasr_bin,
                model_path=self.model_path,
                audio_path=audio_path,
                language=self.language,
            )
        video_id = source.stem
        return parse_transcripts(result.stdout, video_id=video_id)


def extract_audio(video_path: str, audio_path: Path, sample_rate: int) -> None:
    """Extract mono WAV audio with ffmpeg."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise CodeNovaError("Install imageio-ffmpeg before running ASR.") from exc
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-y",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise CodeNovaError(f"ffmpeg audio extraction failed: {result.stderr.strip()[:500]}")


def run_crispasr(
    crispasr_bin: str,
    model_path: str,
    audio_path: Path,
    language: str,
) -> subprocess.CompletedProcess[str]:
    """Run CrispASR with the Qwen3 backend."""
    command = [
        crispasr_bin,
        "--backend",
        "qwen3",
        "-m",
        model_path,
        "-f",
        str(audio_path),
        "-l",
        language,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise CodeNovaError(f"CrispASR failed: {result.stderr.strip()[:500]}")
    return result


def parse_transcripts(raw: str, video_id: str) -> list[Transcript]:
    """Parse SRT-like output, falling back to one untimed transcript."""
    stripped = raw.strip()
    if not stripped:
        return []
    srt_segments = parse_srt(stripped, video_id=video_id)
    if srt_segments:
        return srt_segments
    return [Transcript(video_id=video_id, text=stripped)]


def parse_srt(raw: str, video_id: str) -> list[Transcript]:
    """Parse simple SRT blocks into transcript segments."""
    blocks = re.split(r"\n\s*\n", raw.strip())
    transcripts: list[Transcript] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line_index = 1 if lines[0].isdigit() else 0
        if time_line_index >= len(lines) or "-->" not in lines[time_line_index]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[time_line_index].split("-->", 1)]
        text = " ".join(lines[time_line_index + 1 :]).strip()
        if not text:
            continue
        transcripts.append(
            Transcript(
                video_id=video_id,
                text=text,
                start_time_sec=parse_srt_timestamp(start_raw),
                end_time_sec=parse_srt_timestamp(end_raw),
            )
        )
    return transcripts


def parse_srt_timestamp(value: str) -> float:
    """Convert ``HH:MM:SS,mmm`` or ``HH:MM:SS.mmm`` to seconds."""
    match = re.match(r"^(\d+):(\d{2}):(\d{2})[,.](\d{1,3})", value)
    if not match:
        raise CodeNovaError(f"Invalid SRT timestamp: {value}")
    hours, minutes, seconds, millis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000
