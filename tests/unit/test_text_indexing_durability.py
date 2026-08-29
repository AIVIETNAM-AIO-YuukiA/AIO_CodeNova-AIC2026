import pytest

from config.settings import Experiment, PipelineConfig
from core.types import FrameRecord
from indexing.extract_text import _TextSink, extract_asr, extract_ocr
from indexing.manifest import JsonlManifest
from indexing.state import JobState
from stores.text.base import TextDocument


class FakeTextIndex:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.documents = []

    def index_documents(self, documents):
        if self.fail:
            raise RuntimeError("Elasticsearch unavailable")
        self.documents.extend(documents)


def _experiment(tmp_path):
    return Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))


def test_text_sink_persists_jsonl_before_derived_index_failure(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    monkeypatch.setattr(
        "indexing.extract_text.build_text_index", lambda experiment: FakeTextIndex(fail=True)
    )
    sink = _TextSink(experiment)
    document = TextDocument("d1", "v1", "hello", "ocr", frame_id="f1")

    with pytest.raises(RuntimeError, match="Elasticsearch unavailable"):
        sink.write([document])

    rows = JsonlManifest(tmp_path / "manifests" / "text.jsonl").read_all(strict=True)
    assert [row["doc_id"] for row in rows] == ["d1"]


def test_ocr_no_text_still_writes_empty_text_document(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    (tmp_path / "frame.jpg").write_bytes(b"frame")
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").append(
        FrameRecord("f1", "v1", "s1", "frame.jpg", 1, 0.04).to_dict()
    )

    class FakeOcr:
        def recognize(self, path):
            return ""

    monkeypatch.setattr("indexing.extract_text.VllmOcrModel", FakeOcr)
    fake_index = FakeTextIndex()
    monkeypatch.setattr("indexing.extract_text.build_text_index", lambda experiment: fake_index)

    assert extract_ocr(experiment) == 1
    assert JobState(tmp_path / "jobs.sqlite").get_status("f1", "EXTRACT_OCR") == "COMPLETED"
    assert [doc.frame_id for doc in fake_index.documents] == ["f1"]
    assert fake_index.documents[0].text == ""
    assert extract_ocr(experiment) == 0


def test_asr_no_audio_is_completed_without_output(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    JsonlManifest(tmp_path / "manifests" / "videos.jsonl").append(
        {"video_id": "v1", "path": "video.mp4", "checksum": "x", "size_bytes": 1}
    )

    class FakeAsr:
        def transcribe(self, path):
            return []

    monkeypatch.setattr("indexing.extract_text.GipformerAsrModel", FakeAsr)
    monkeypatch.setattr(
        "indexing.extract_text.build_text_index", lambda experiment: FakeTextIndex()
    )

    assert extract_asr(experiment) == 0
    assert (
        JobState(tmp_path / "jobs.sqlite").get_status("v1", "EXTRACT_ASR") == "COMPLETED_NO_OUTPUT"
    )
