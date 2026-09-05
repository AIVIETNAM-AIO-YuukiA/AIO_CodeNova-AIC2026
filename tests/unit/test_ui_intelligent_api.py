from http import HTTPStatus

from config.settings import Experiment, PipelineConfig
from indexing.manifest import JsonlManifest
from ui import api as ui_api


def _experiment(tmp_path) -> Experiment:
    return Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))


def test_intelligent_api_forwards_new_controls_and_defaults(monkeypatch, tmp_path) -> None:
    experiment = _experiment(tmp_path)
    captured = {}

    def fake_search(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"results": [{"frame_id": "f1", "frame_path": "frames/f1.jpg"}]}

    monkeypatch.setattr(ui_api, "intelligent_search", fake_search)

    response = ui_api.handle_intelligent_search(
        {
            "query": "người nói cạnh biển hiệu",
            "enabled_models": ["jina-clip-v2"],
            "use_llm": False,
            "use_evidence_reranker": True,
        },
        experiment,
        default_top_k=20,
    )

    assert captured["args"] == (experiment,)
    assert captured["kwargs"] == {
        "query": "người nói cạnh biển hiệu",
        "top_k": 20,
        "enable_kis": True,
        "enable_ocr": True,
        "enable_asr": True,
        "enabled_models": ["jina-clip-v2"],
        "use_reranker": None,
        "use_llm": False,
        "fusion_mode": "adaptive",
        "text_search_mode": "separate",
        "temporal_asr": True,
        "use_evidence_reranker": True,
        "max_frames_per_shot": 2,
    }
    assert response["results"][0]["image_url"] == "/frame?path=frames/f1.jpg"


def test_video_shots_uses_asr_intervals_instead_of_exact_timestamp(monkeypatch, tmp_path) -> None:
    experiment = _experiment(tmp_path)
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").extend(
        [
            {
                "frame_id": "v1_s000001_f00000300",
                "video_id": "v1",
                "shot_id": "v1_s000001",
                "frame_path": "frames/v1/f300.jpg",
                "frame_index": 300,
                "timestamp_sec": 10.0,
            },
            {
                "frame_id": "v1_s000002_f00000513",
                "video_id": "v1",
                "shot_id": "v1_s000002",
                "frame_path": "frames/v1/f513.jpg",
                "frame_index": 513,
                "timestamp_sec": 17.1,
            },
        ]
    )
    JsonlManifest(tmp_path / "manifests" / "text.jsonl").extend(
        [
            {
                "doc_id": "v1__asr__0000",
                "video_id": "v1",
                "source": "asr",
                "text": "sụt lún đồng bằng sông cửu long",
                "timestamp_sec": 5.0,
                "frame_id": None,
            },
            {
                "doc_id": "v1__asr__0001",
                "video_id": "v1",
                "source": "asr",
                "text": "bản tin tiếp theo",
                "timestamp_sec": 15.0,
                "frame_id": None,
            },
        ]
    )

    monkeypatch.setattr(ui_api, "_VIDEO_NAME_CACHE", {})
    monkeypatch.setattr(ui_api, "_CAPTIONS_CACHE", {})
    monkeypatch.setattr(ui_api, "_TEXT_CACHE", [])
    monkeypatch.setattr(ui_api, "_FRAMES_BY_VIDEO_CACHE", {})
    monkeypatch.setattr(ui_api, "_VIDEO_TEXT_CACHE", {})

    payload, status = ui_api.handle_video_shots({"video_id": ["v1"]}, experiment)

    assert status == HTTPStatus.OK
    frame = payload["shots"][0]["frames"][0]
    assert frame["timestamp_sec"] == 10.0
    assert frame["asr"] == "sụt lún đồng bằng sông cửu long"


def test_video_shots_uses_same_nearest_fallback_as_search(monkeypatch, tmp_path) -> None:
    experiment = _experiment(tmp_path)
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").append(
        {
            "frame_id": "v1_s000001_f00000300",
            "video_id": "v1",
            "shot_id": "v1_s000001",
            "frame_path": "frames/v1/f300.jpg",
            "frame_index": 300,
            "timestamp_sec": 10.0,
        }
    )
    JsonlManifest(tmp_path / "manifests" / "text.jsonl").append(
        {
            "doc_id": "v1__asr__0000",
            "video_id": "v1",
            "source": "asr",
            "text": "đoạn thoại ở vùng không có keyframe",
            "timestamp_sec": 100.0,
            "frame_id": None,
        }
    )
    monkeypatch.setattr(ui_api, "_VIDEO_NAME_CACHE", {})
    monkeypatch.setattr(ui_api, "_CAPTIONS_CACHE", {})
    monkeypatch.setattr(ui_api, "_TEXT_CACHE", [])
    monkeypatch.setattr(ui_api, "_FRAMES_BY_VIDEO_CACHE", {})
    monkeypatch.setattr(ui_api, "_VIDEO_TEXT_CACHE", {})

    payload, status = ui_api.handle_video_shots({"video_id": ["v1"]}, experiment)

    assert status == HTTPStatus.OK
    assert payload["shots"][0]["frames"][0]["asr"] == ("đoạn thoại ở vùng không có keyframe")
