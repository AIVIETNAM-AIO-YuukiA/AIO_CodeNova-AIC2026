from types import SimpleNamespace
from unittest.mock import MagicMock

from config.settings import Experiment, PipelineConfig
from core.types import SearchResult
from indexing.manifest import JsonlManifest
from retrieval import intelligent_search as intelligent_module
from retrieval.intelligent_search import (
    _adaptive_weight,
    _diversify_by_shot,
    _document_quality,
    _evidence_rerank,
    _fixed_weight,
    _search_text,
    intelligent_search,
)


def _processed(**overrides):
    values = {
        "raw_query": "người đi xe máy",
        "visual_prompt": "a person riding a motorbike",
        "visual_prompt_vi": "người đi xe máy",
        "caption_keywords": [],
        "ocr_keywords": [],
        "asr_keywords": [],
        "normalized_keywords": {"caption": [], "ocr": [], "asr": []},
        "metadata": {},
        "weights": {"caption_bonus": 0.0, "ocr_bonus": 0.0, "asr_bonus": 0.0},
        "routing_mode": "heuristic",
        "modality_confidence": {"kis": 1.0, "ocr": 0.0, "asr": 0.0, "caption": 0.0},
        "llm_status": "disabled",
        "fallback_reason": None,
        "llm_calls": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _IdentityHydrator:
    def __init__(self, experiment):
        pass

    def hydrate(self, results):
        return results


def test_intelligent_search_processes_query_exactly_once(monkeypatch, tmp_path) -> None:
    experiment = Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))
    retriever = MagicMock()
    retriever.query_processor.process.return_value = _processed()
    retriever.search_processed.return_value = [
        SearchResult("f1", "v1", 0.9, shot_id="s1", timestamp_sec=1.0)
    ]
    monkeypatch.setattr(intelligent_module, "_get_retriever", lambda _: retriever)
    monkeypatch.setattr(intelligent_module, "ResultHydrator", _IdentityHydrator)

    response = intelligent_search(
        experiment,
        "người đi xe máy",
        enable_ocr=False,
        enable_asr=False,
        enable_caption=False,
        use_llm=False,
    )

    retriever.query_processor.process.assert_called_once_with(
        "người đi xe máy", enabled_models=None, use_llm=False
    )
    retriever.search_processed.assert_called_once()
    assert response["results"][0]["frame_id"] == "f1"
    assert response["component_status"]["kis"] == "used"
    assert response["analysis"]["llm_calls"] == 0
    assert "timing_ms" in response


def test_no_text_hits_leave_kis_ranking_unchanged(monkeypatch, tmp_path) -> None:
    experiment = Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))
    retriever = MagicMock()
    retriever.query_processor.process.return_value = _processed(
        ocr_keywords=["giảm giá"],
        normalized_keywords={"caption": [], "ocr": ["giam gia"], "asr": []},
        modality_confidence={"kis": 1.0, "ocr": 1.0, "asr": 0.0, "caption": 0.0},
        weights={"caption_bonus": 0.0, "ocr_bonus": 0.3, "asr_bonus": 0.0},
    )
    retriever.search_processed.return_value = [
        SearchResult("f1", "v1", 0.9, shot_id="s1"),
        SearchResult("f2", "v1", 0.8, shot_id="s2"),
    ]
    text_index = MagicMock()
    text_index.search_documents.return_value = []
    monkeypatch.setattr(intelligent_module, "_get_retriever", lambda _: retriever)
    monkeypatch.setattr(intelligent_module, "ResultHydrator", _IdentityHydrator)
    monkeypatch.setattr(intelligent_module, "build_text_index", lambda _: text_index)

    response = intelligent_search(
        experiment,
        "biển giảm giá",
        enable_asr=False,
        enable_caption=False,
        use_evidence_reranker=False,
    )

    assert [row["frame_id"] for row in response["results"]] == ["f1", "f2"]
    assert response["component_status"]["ocr"] == "no_hits"
    assert response["fusion_weights"]["ocr"] == 0.0


def test_text_backend_failure_preserves_kis_results(monkeypatch, tmp_path) -> None:
    experiment = Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))
    retriever = MagicMock()
    retriever.query_processor.process.return_value = _processed(
        caption_keywords=["cửa hàng"],
        normalized_keywords={"caption": ["cua hang"], "ocr": [], "asr": []},
        modality_confidence={"kis": 1.0, "ocr": 0.0, "asr": 0.0, "caption": 0.8},
        weights={"caption_bonus": 0.4, "ocr_bonus": 0.0, "asr_bonus": 0.0},
    )
    retriever.search_processed.return_value = [
        SearchResult("f1", "v1", 0.9, shot_id="s1")
    ]
    text_index = MagicMock()
    text_index.search_documents.side_effect = ConnectionError("elasticsearch down")
    monkeypatch.setattr(intelligent_module, "_get_retriever", lambda _: retriever)
    monkeypatch.setattr(intelligent_module, "ResultHydrator", _IdentityHydrator)
    monkeypatch.setattr(intelligent_module, "build_text_index", lambda _: text_index)

    response = intelligent_search(experiment, "người trước cửa hàng")

    assert [row["frame_id"] for row in response["results"]] == ["f1"]
    assert response["component_status"]["caption"] == "error"
    assert response["fusion_weights"]["caption"] == 0.0
    assert "elasticsearch down" in response["component_stats"]["caption"]["error"]


def test_adaptive_weight_uses_confidence_and_hit_quality() -> None:
    weight = _adaptive_weight(0.8, {"hit_quality": 0.5})
    assert weight == 0.4
    assert _adaptive_weight(1.0, {"hit_quality": 1.0}) == 0.5


def test_fixed_weight_is_query_independent() -> None:
    assert _fixed_weight("ocr") == 0.3
    assert _fixed_weight("asr") == 0.3
    assert _fixed_weight("caption") == 0.3


def test_evidence_reranker_respects_small_adaptive_weight() -> None:
    results = [
        SearchResult("visual", "v", 1.0),
        SearchResult("weak_ocr", "v", 0.96),
    ]
    reranked = _evidence_rerank(
        results,
        {"weak_ocr": {"ocr": {"rank": 1, "raw_score": 5.0, "normalized_score": 1.0}}},
        {"kis": 1.0, "ocr": 0.01},
        {
            "weak_ocr": [
                {
                    "source": "ocr",
                    "document_quality": 1.0,
                    "temporal_weight": 1.0,
                    "exact_phrase": True,
                }
            ]
        },
        limit=30,
    )

    assert [row.frame_id for row in reranked] == ["visual", "weak_ocr"]


def test_ocr_aliases_and_near_duplicate_frames_count_once(monkeypatch, tmp_path) -> None:
    experiment = Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").extend(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "shot_id": "s1",
                "frame_path": "frames/v1/f1.jpg",
                "frame_index": 25,
                "timestamp_sec": 1.9,
            },
            {
                "frame_id": "f2",
                "video_id": "v1",
                "shot_id": "s1",
                "frame_path": "frames/v1/f2.jpg",
                "frame_index": 30,
                "timestamp_sec": 2.1,
            },
        ]
    )
    documents = [
        {
            "doc_id": frame_id + "__ocr",
            "frame_id": frame_id,
            "video_id": "v1",
            "source": "ocr",
            "text": "GIẢM GIÁ 50%",
            "timestamp_sec": timestamp,
            "score": 5.0,
        }
        for frame_id, timestamp in (("f1", 1.9), ("f2", 2.1))
    ]
    text_index = MagicMock()
    text_index.search_documents.return_value = documents
    monkeypatch.setattr(intelligent_module, "build_text_index", lambda _: text_index)

    outcome = _search_text(
        experiment,
        ["GIẢM GIÁ", "giam gia"],
        source="ocr",
        top_k=20,
    )

    assert len(outcome.results) == 1
    assert outcome.stats["keyword_count"] == 1
    assert outcome.stats["query_variant_count"] == 2
    assert outcome.stats["keyword_coverage"] == 1.0


def test_ocr_quality_rejects_symbol_noise() -> None:
    assert _document_quality("--- !!!", "ocr") == 0.0
    assert _document_quality("GIẢM GIÁ 50%", "ocr") > 0.7


def test_shot_diversity_prefers_new_shots_then_fills() -> None:
    results = [
        SearchResult("f1", "v", 1.0, shot_id="s1"),
        SearchResult("f2", "v", 0.9, shot_id="s1"),
        SearchResult("f3", "v", 0.8, shot_id="s2"),
        SearchResult("f4", "v", 0.7, shot_id="s3"),
    ]

    diversified = _diversify_by_shot(results, top_k=3, limit=1)

    assert [row.frame_id for row in diversified] == ["f1", "f3", "f4"]
