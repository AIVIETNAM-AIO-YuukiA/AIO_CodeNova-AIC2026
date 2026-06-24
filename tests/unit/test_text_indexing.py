from dataclasses import dataclass

from config.settings import Experiment, PipelineConfig
from indexing.manifest import JsonlManifest
from indexing.text import build_text_index_from_artifacts, load_text_documents, run_asr, run_ocr
from modules.asr.base import Transcript


@dataclass
class FakeOcrModel:
    text: str = "EXIT\nA12"

    def recognize(self, frame_path: str) -> str:
        assert frame_path.endswith(".jpg")
        return self.text


@dataclass
class FakeAsrModel:
    def transcribe(self, video_path: str) -> list[Transcript]:
        assert video_path.endswith(".mp4")
        return [
            Transcript(
                video_id="ignored",
                text="hello from the audio",
                start_time_sec=1.0,
                end_time_sec=2.5,
            )
        ]


def make_experiment(tmp_path) -> Experiment:
    return Experiment(name="exp1", run_dir=tmp_path, config=PipelineConfig(runs_dir=tmp_path))


def test_run_ocr_writes_artifact_records(tmp_path) -> None:
    experiment = make_experiment(tmp_path)
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").append(
        {
            "frame_id": "frame1",
            "video_id": "video1",
            "shot_id": "shot1",
            "frame_path": "frames/frame1.jpg",
            "frame_index": 42,
            "timestamp_sec": 1.4,
        }
    )

    count = run_ocr(experiment, model=FakeOcrModel())

    rows = JsonlManifest(tmp_path / "text" / "ocr.jsonl").read_all()
    assert count == 1
    assert rows[0]["frame_id"] == "frame1"
    assert rows[0]["text"] == "EXIT\nA12"


def test_run_asr_writes_artifact_records_with_manifest_video_id(tmp_path) -> None:
    experiment = make_experiment(tmp_path)
    JsonlManifest(tmp_path / "manifests" / "videos.jsonl").append(
        {
            "video_id": "video1",
            "path": "data/sample.mp4",
            "checksum": "abc",
            "size_bytes": 123,
        }
    )

    count = run_asr(experiment, model=FakeAsrModel())

    rows = JsonlManifest(tmp_path / "text" / "asr.jsonl").read_all()
    assert count == 1
    assert rows[0]["video_id"] == "video1"
    assert rows[0]["text"] == "hello from the audio"
    assert rows[0]["start_time_sec"] == 1.0


def test_load_text_documents_converts_ocr_and_asr_artifacts(tmp_path) -> None:
    experiment = make_experiment(tmp_path)
    JsonlManifest(tmp_path / "text" / "ocr.jsonl").append(
        {
            "frame_id": "frame1",
            "video_id": "video1",
            "timestamp_sec": 1.4,
            "text": "street sign",
        }
    )
    JsonlManifest(tmp_path / "text" / "asr.jsonl").append(
        {
            "video_id": "video1",
            "text": "spoken words",
            "start_time_sec": 2.0,
            "end_time_sec": 3.0,
        }
    )

    documents = load_text_documents(experiment)

    assert [doc.doc_id for doc in documents] == ["ocr:frame1", "asr:video1:2.0"]
    assert [doc.source for doc in documents] == ["ocr", "asr"]
    assert documents[0].frame_id == "frame1"
    assert documents[1].frame_id is None


def test_build_text_index_from_artifacts_indexes_documents(monkeypatch, tmp_path) -> None:
    experiment = make_experiment(tmp_path)
    JsonlManifest(tmp_path / "text" / "ocr.jsonl").append(
        {
            "frame_id": "frame1",
            "video_id": "video1",
            "timestamp_sec": 1.4,
            "text": "street sign",
        }
    )
    indexed_batches = []

    class FakeTextIndex:
        def index_documents(self, documents):
            indexed_batches.append(documents)

    monkeypatch.setattr("indexing.text.build_text_index", lambda experiment: FakeTextIndex())

    count = build_text_index_from_artifacts(experiment)

    assert count == 1
    assert indexed_batches[0][0].doc_id == "ocr:frame1"
