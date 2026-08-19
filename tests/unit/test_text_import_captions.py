from cli.main import build_parser
from config.settings import Experiment, PipelineConfig
from indexing.extract_text import import_text
from indexing.manifest import JsonlManifest


class FakeTextIndex:
    def __init__(self) -> None:
        self.documents = {}

    def index_documents(self, documents) -> None:
        self.documents.update({document.doc_id: document for document in documents})


def _experiment(tmp_path):
    return Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))


def _write_manifests(tmp_path) -> None:
    JsonlManifest(tmp_path / "manifests" / "text.jsonl").extend(
        [
            {
                "doc_id": "f1__ocr",
                "video_id": "v1",
                "frame_id": "f1",
                "source": "ocr",
                "text": "BẢNG HIỆU",
                "timestamp_sec": 1.0,
            },
            {
                "doc_id": "empty__caption",
                "video_id": "v1",
                "frame_id": "f1",
                "source": "caption",
                "text": "  ",
                "timestamp_sec": 1.0,
            },
            {
                "doc_id": "f1__caption",
                "video_id": "v1",
                "frame_id": "f1",
                "source": "caption",
                "text": "Caption đã có trong text export.",
                "timestamp_sec": 1.0,
            },
        ]
    )
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").append(
        {
            "frame_id": "f1",
            "video_id": "v1",
            "shot_id": "s1",
            "frame_path": "frames/v1/f1.jpg",
            "frame_index": 25,
            "timestamp_sec": 1.0,
        }
    )
    captions = JsonlManifest(tmp_path / "manifests" / "captions.jsonl")
    captions.extend(
        [
            {"frame_id": "f1", "caption": "  Một người đứng trước cửa hàng.  "},
            {"frame_id": "f2", "caption": "không có frame tương ứng"},
            {"frame_id": "f1", "caption": "   "},
            {"frame_id": "f1", "caption": None},
        ]
    )


def test_import_text_includes_valid_captions_and_is_idempotent(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    _write_manifests(tmp_path)
    index = FakeTextIndex()
    monkeypatch.setattr("indexing.extract_text.build_text_index", lambda experiment: index)

    assert import_text(experiment) == 2
    assert import_text(experiment) == 2

    assert set(index.documents) == {"f1__ocr", "f1__caption"}
    assert index.documents["f1__caption"].source == "caption"
    assert index.documents["f1__caption"].text == "Một người đứng trước cửa hàng."


def test_import_text_can_skip_captions(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    _write_manifests(tmp_path)
    index = FakeTextIndex()
    monkeypatch.setattr("indexing.extract_text.build_text_index", lambda experiment: index)

    assert import_text(experiment, include_captions=False) == 1
    assert set(index.documents) == {"f1__ocr"}


def test_import_text_supports_a_caption_only_artifact(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").append(
        {
            "frame_id": "f1",
            "video_id": "v1",
            "shot_id": "s1",
            "frame_path": "frames/v1/f1.jpg",
            "frame_index": 25,
            "timestamp_sec": 1.0,
        }
    )
    JsonlManifest(tmp_path / "manifests" / "captions.jsonl").append(
        {"frame_id": "f1", "caption": "Một người đứng trước cửa hàng."}
    )
    index = FakeTextIndex()
    monkeypatch.setattr("indexing.extract_text.build_text_index", lambda experiment: index)

    assert import_text(experiment) == 1
    assert set(index.documents) == {"f1__caption"}


def test_import_text_parser_enables_captions_by_default_and_supports_opt_out():
    parser = build_parser()

    default = parser.parse_args(["import-text", "--experiment-name", "result"])
    disabled = parser.parse_args(
        ["import-text", "--experiment-name", "result", "--no-captions"]
    )

    assert default.no_captions is False
    assert disabled.no_captions is True
