import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from core.errors import CodeNovaError
from modules.asr.factory import build_asr_model
from modules.asr.qwen3_gguf import Qwen3GgufAsrModel, parse_srt, parse_transcripts, run_crispasr
from modules.ocr.factory import build_ocr_model
from modules.ocr.gemini import GeminiOcrModel


def test_ocr_factory_requires_gemini_key(monkeypatch) -> None:
    monkeypatch.setenv("OCR_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(CodeNovaError, match="GEMINI_API_KEY"):
        build_ocr_model()


def test_asr_factory_builds_qwen3_backend(monkeypatch) -> None:
    monkeypatch.setenv("ASR_BACKEND", "qwen3_gguf")
    monkeypatch.setenv("ASR_QWEN_MODEL_PATH", "models/model.gguf")
    monkeypatch.setenv("ASR_CRISPASR_BIN", "external/crispasr")
    monkeypatch.setenv("ASR_LANGUAGE", "vi")
    monkeypatch.setenv("ASR_AUDIO_SAMPLE_RATE", "8000")

    model = build_asr_model()

    assert isinstance(model, Qwen3GgufAsrModel)
    assert model.model_path == "models/model.gguf"
    assert model.crispasr_bin == "external/crispasr"
    assert model.language == "vi"
    assert model.sample_rate == 8000


def test_gemini_ocr_uses_inline_image(monkeypatch, tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image")
    response = MagicMock(text="HELLO")
    client = MagicMock()
    client.models.generate_content.return_value = response
    genai_mock = ModuleType("google.genai")
    genai_mock.Client = MagicMock(return_value=client)
    google_mock = ModuleType("google")
    google_mock.genai = genai_mock
    monkeypatch.setitem(sys.modules, "google", google_mock)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mock)

    model = GeminiOcrModel(api_key="key", model_name="gemini-test")

    assert model.recognize(str(image)) == "HELLO"
    client.models.generate_content.assert_called_once()
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-test"
    assert kwargs["contents"][1]["inline_data"]["data"] == b"fake image"


def test_run_crispasr_builds_expected_command(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(command, capture_output, text):
        calls.append((command, capture_output, text))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_crispasr("crispasr", "model.gguf", tmp_path / "audio.wav", "auto")

    assert result.stdout == "ok"
    assert calls[0][0] == [
        "crispasr",
        "--backend",
        "qwen3",
        "-m",
        "model.gguf",
        "-f",
        str(tmp_path / "audio.wav"),
        "-l",
        "auto",
    ]


def test_parse_srt_segments() -> None:
    raw = """1
00:00:01,000 --> 00:00:02,500
hello there

2
00:00:03.000 --> 00:00:04.250
second line
"""

    transcripts = parse_srt(raw, video_id="video1")

    assert len(transcripts) == 2
    assert transcripts[0].text == "hello there"
    assert transcripts[0].start_time_sec == 1.0
    assert transcripts[0].end_time_sec == 2.5
    assert transcripts[1].start_time_sec == 3.0
    assert transcripts[1].end_time_sec == 4.25


def test_parse_transcripts_falls_back_to_plain_text() -> None:
    transcripts = parse_transcripts("plain transcript", video_id="video1")

    assert len(transcripts) == 1
    assert transcripts[0].text == "plain transcript"
    assert transcripts[0].start_time_sec is None
