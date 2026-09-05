import json

import pytest

from evaluation.intelligent import (
    QueryQrel,
    RelevantMoment,
    evaluate_ablation,
    evaluate_ranking,
    interval_iou,
    load_qrels,
)


def _qrel() -> QueryQrel:
    return QueryQrel(
        query_id="q1",
        query="spoken phrase",
        relevant_frame_ids=frozenset({"f2"}),
        relevant_moments=(RelevantMoment("v1", 9.0, 12.0),),
        expected_modalities=frozenset({"asr"}),
        group="asr-only",
    )


def test_ranking_metrics_and_temporal_iou() -> None:
    metrics = evaluate_ranking(
        _qrel(),
        [
            {"frame_id": "f1", "video_id": "v0", "timestamp_sec": 1.0},
            {"frame_id": "f2", "video_id": "v1", "timestamp_sec": 10.0},
        ],
        [1, 2],
    )

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg@2"] > 0.0
    assert metrics["temporal_iou"] > 0.0


def test_interval_iou() -> None:
    assert interval_iou((0.0, 2.0), (1.0, 3.0)) == pytest.approx(1.0 / 3.0)


def test_moment_only_qrels_contribute_to_ranking_metrics() -> None:
    qrel = QueryQrel(
        query_id="temporal",
        query="đoạn bản tin",
        relevant_frame_ids=frozenset(),
        relevant_moments=(RelevantMoment("v1", 9.0, 12.0),),
        expected_modalities=frozenset({"asr"}),
    )

    metrics = evaluate_ranking(
        qrel,
        [{"frame_id": "f10", "video_id": "v1", "timestamp_sec": 10.0}],
        [1],
    )

    assert metrics["recall@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_load_qrels_rejects_unlabelled_rows(tmp_path) -> None:
    path = tmp_path / "qrels.jsonl"
    path.write_text(json.dumps({"query_id": "q", "query": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no relevance labels"):
        load_qrels(path)


def test_evaluate_ablation_collects_routing_and_latency() -> None:
    def fake_search(**kwargs):
        assert kwargs["query"] == "spoken phrase"
        return {
            "results": [{"frame_id": "f2", "video_id": "v1", "timestamp_sec": 10.0}],
            "analysis": {"routing_mode": "llm", "llm_attempts": 1},
            "component_status": {"kis": "used", "asr": "used"},
        }

    report = evaluate_ablation([_qrel()], fake_search, {}, [1])

    assert report["summary"]["recall@1"] == 1.0
    assert report["summary"]["route_recall"] == 1.0
    assert report["summary"]["queries"] == 1
    assert report["groups"]["asr-only"]["queries"] == 1
