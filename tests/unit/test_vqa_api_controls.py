"""Focused API/UI contract tests for grounded multi-frame VQA."""

from __future__ import annotations

import pytest

from api.app import _render_index_html
from api.schemas.vqa import VqaSearchRequest
from api.services import vqa_service
from config.settings import Experiment, PipelineConfig
from ui import api as legacy_ui_api
from ui.views.sidebar import SIDEBAR_HTML
from ui.views.scripts import APP_JS


def _experiment(tmp_path) -> Experiment:
    return Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))


def test_vqa_request_defaults_to_grounded_pipeline() -> None:
    request = VqaSearchRequest()

    assert request.enabled_models is None
    assert request.use_reranker is None
    assert request.use_llm is True
    assert request.pipeline_mode == "grounded"


def test_vqa_request_rejects_unknown_pipeline_mode() -> None:
    with pytest.raises(ValueError):
        VqaSearchRequest(pipeline_mode="experimental")


def test_vqa_request_rejects_invalid_limits_and_empty_model_selection() -> None:
    with pytest.raises(ValueError):
        VqaSearchRequest(top_k=0)
    with pytest.raises(ValueError):
        VqaSearchRequest(top_k=101)
    with pytest.raises(ValueError):
        VqaSearchRequest(reranker_top_k=-1)
    with pytest.raises(ValueError):
        VqaSearchRequest(enabled_models=[])


def test_fastapi_service_forwards_grounded_controls_and_nested_frame_urls(monkeypatch) -> None:
    captured = {}
    reranker = object()

    def fake_vqa_search(**kwargs):
        captured.update(kwargs)
        return {
            "results": [{"frame_path": "runs/exp/frames/result.jpg"}],
            "display_results": [
                {"frame_path": "runs/exp/frames/display-candidate.jpg"}
            ],
            "evidence_frames": [{"frame_path": "runs/exp/frames/evidence.jpg"}],
            "selected_candidate": {
                "evidence_frames": [{"frame_path": "runs/exp/frames/selected.jpg"}]
            },
            "candidate_answers": [
                {"evidence_frames": [{"frame_path": "runs/exp/frames/candidate.jpg"}]}
            ],
        }

    monkeypatch.setattr(vqa_service, "vqa_search", fake_vqa_search)
    request = VqaSearchRequest(
        query="four items on a plate",
        question="What is X?",
        top_k=50,
        enabled_models=["siglip2-so400m"],
        use_reranker=True,
        use_llm=False,
        pipeline_mode="grounded",
    )

    response = vqa_service.run_vqa_search(
        experiment=object(),
        default_top_k=20,
        reranker=reranker,
        reranker_top_k=10,
        req=request,
    )

    assert captured["top_k"] == 50
    assert captured["enabled_models"] == ["siglip2-so400m"]
    assert captured["use_reranker"] is True
    assert captured["reranker"] is reranker
    assert captured["use_llm"] is False
    assert captured["pipeline_mode"] == "grounded"
    assert response["results"][0]["image_url"].endswith("runs/exp/frames/result.jpg")
    assert response["display_results"][0]["image_url"].endswith(
        "runs/exp/frames/display-candidate.jpg"
    )
    assert response["evidence_frames"][0]["image_url"].endswith(
        "runs/exp/frames/evidence.jpg"
    )
    assert response["selected_candidate"]["evidence_frames"][0]["image_url"].endswith(
        "runs/exp/frames/selected.jpg"
    )
    assert response["candidate_answers"][0]["evidence_frames"][0]["image_url"].endswith(
        "runs/exp/frames/candidate.jpg"
    )


def test_fastapi_service_honors_disabled_reranker(monkeypatch) -> None:
    captured = {}

    def fake_vqa_search(**kwargs):
        captured.update(kwargs)
        return {"results": []}

    monkeypatch.setattr(vqa_service, "vqa_search", fake_vqa_search)
    request = VqaSearchRequest(use_reranker=False)

    vqa_service.run_vqa_search(object(), 20, object(), 10, request)

    assert captured["use_reranker"] is False
    assert captured["reranker"] is None


def test_legacy_ui_handler_forwards_grounded_controls(monkeypatch, tmp_path) -> None:
    captured = {}
    reranker = object()

    def fake_vqa_search(**kwargs):
        captured.update(kwargs)
        return {"evidence_frames": [{"frame_path": "runs/exp/frames/f1.jpg"}]}

    monkeypatch.setattr(legacy_ui_api, "vqa_search", fake_vqa_search)
    response = legacy_ui_api.handle_vqa_search(
        {
            "query": "query",
            "question": "question",
            "enabled_models": ["jina-clip-v2"],
            "use_reranker": True,
            "use_llm": False,
            "pipeline_mode": "grounded",
        },
        _experiment(tmp_path),
        default_top_k=20,
        reranker=reranker,
        reranker_top_k=10,
    )

    assert captured["enabled_models"] == ["jina-clip-v2"]
    assert captured["use_reranker"] is True
    assert captured["reranker"] is reranker
    assert captured["use_llm"] is False
    assert captured["pipeline_mode"] == "grounded"
    assert response["evidence_frames"][0]["image_url"].endswith("runs/exp/frames/f1.jpg")


def test_legacy_ui_handler_rejects_string_booleans(tmp_path) -> None:
    with pytest.raises(ValueError, match="use_llm must be a boolean"):
        legacy_ui_api.handle_vqa_search(
            {"query": "query", "question": "question", "use_llm": "false"},
            _experiment(tmp_path),
            default_top_k=20,
            reranker=None,
            reranker_top_k=10,
        )


def test_vqa_frontend_sends_and_renders_grounded_contract() -> None:
    assert 'pipeline_mode: eid("vqa-pipeline-mode").value' in APP_JS
    assert 'id="vqa-pipeline-mode"' in SIDEBAR_HTML
    assert '<option value="legacy">' in SIDEBAR_HTML
    assert "renderVqaResponse(data)" in APP_JS
    assert "data.display_results || data.results" in APP_JS
    assert "Grounded multi-frame VQA" in APP_JS
    assert "Evidence frames" in APP_JS
    assert "Candidate verification" in APP_JS
    assert "Ordered-event Retrieval" in APP_JS
    assert "frame.evidence_label" in APP_JS
    assert "Verifier error:" in APP_JS
    assert "Select at least one embedding model." in APP_JS


def test_video_name_map_handles_windows_paths_on_linux(tmp_path) -> None:
    experiment = _experiment(tmp_path)
    manifests = experiment.run_dir / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "videos.jsonl").write_text(
        '{"video_id":"target","path":"data\\\\raw_videos\\\\L26_V254.mp4"}\n',
        encoding="utf-8",
    )
    legacy_ui_api._VIDEO_NAME_CACHE.clear()

    names = legacy_ui_api._get_video_name_map(experiment)

    assert names["target"] == "L26_V254.mp4"


def test_fastapi_index_renders_active_model_names(tmp_path) -> None:
    experiment = Experiment(
        "exp",
        tmp_path,
        PipelineConfig(
            runs_dir=tmp_path,
            embedding_models=("jina-clip-v2", "siglip2-so400m"),
        ),
    )

    html = _render_index_html(experiment, default_top_k=25)

    assert "__ACTIVE_MODELS__" not in html
    assert "__MODEL_CHECKBOXES__" not in html
    assert "Models: jina-clip-v2, siglip2-so400m" in html
    assert 'name="model_jina-clip-v2"' in html
    assert 'name="model_siglip2-so400m"' in html
    assert 'name="model_vietnamese-embedding"' not in html
