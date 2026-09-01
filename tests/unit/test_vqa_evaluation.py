import json
from pathlib import Path

import pytest

from evaluation.vqa import (
    RelevantMoment,
    RelevantVideo,
    VqaQrel,
    answer_exact_match,
    answer_token_f1,
    evaluate_queries,
    evaluate_response,
    interval_iou,
    load_qrels,
    normalize_answer,
)


def _qrel() -> VqaQrel:
    return VqaQrel(
        query_id="q1",
        query="đặt bốn con X lên đĩa rồi cầm hai con X",
        question="X là con gì?",
        context="",
        acceptable_answers=("nghêu", "con nghêu"),
        relevant_videos=(RelevantVideo(video_name="L26_V254.mp4"),),
        relevant_frame_ids=frozenset({"frame-answer"}),
        relevant_moments=(
            RelevantMoment(
                video_name="L26_V254.mp4",
                start_sec=10.0,
                end_sec=20.0,
            ),
        ),
        group="object-sequence",
    )


def test_answer_normalization_and_metrics() -> None:
    assert normalize_answer("X là CON NGHÊU!") == "con nghêu"
    assert answer_exact_match("Đáp án là con nghêu.", ("con nghêu",)) == 1.0
    assert answer_token_f1("Tôi nhìn thấy con nghêu", ("con nghêu",)) == pytest.approx(4 / 7)


def test_load_qrels_accepts_video_and_answer_without_fabricated_moment(tmp_path) -> None:
    path = tmp_path / "vqa.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "scene",
                "question": "what?",
                "acceptable_answers": ["nghêu"],
                "relevant_videos": [{"video_name": "L26_V254.mp4"}],
                "relevant_frame_ids": [],
                "relevant_moments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    qrel = load_qrels(path)[0]

    assert qrel.acceptable_answers == ("nghêu",)
    assert qrel.relevant_videos[0].video_name == "L26_V254.mp4"
    assert qrel.relevant_moments == ()


def test_example_qrel_keeps_l26_v254_ngheu_regression() -> None:
    path = Path(__file__).resolve().parents[2] / "eval" / "vqa_qrels.example.jsonl"

    qrel = load_qrels(path)[0]

    assert qrel.query_id == "vqa_l26_v254_ingredient"
    assert "nghêu" in qrel.acceptable_answers
    assert qrel.relevant_videos == (RelevantVideo(video_name="L26_V254.mp4"),)


def test_evaluate_response_scores_video_answer_evidence_and_temporal_iou() -> None:
    response = {
        "answer": "X là con nghêu.",
        "answer_status": "answered",
        "answer_confidence": 0.9,
        "selected_candidate": {
            "video_name": "data\\raw_videos\\L26_V254.mp4",
            "start_sec": 12.0,
            "end_sec": 18.0,
        },
        "candidate_answers": [
            {"candidate": {"video_name": "L26_V254.mp4"}},
            {"candidate": {"video_name": "L26_V999.mp4"}},
        ],
        "evidence_frames": [
            {
                "frame_id": "frame-answer",
                "video_name": "L26_V254.mp4",
                "timestamp_sec": 14.0,
            }
        ],
        "usage": {"openrouter_calls": 4, "cost": 0.012},
    }

    metrics = evaluate_response(_qrel(), response)

    assert metrics["answer_em"] == 1.0
    assert metrics["answer_f1"] == 1.0
    assert metrics["video_recall@3"] == 1.0
    assert metrics["evidence_accuracy"] == 1.0
    assert metrics["temporal_iou"] == pytest.approx(0.6)
    assert metrics["openrouter_calls"] == 4
    assert metrics["openrouter_cost"] == pytest.approx(0.012)


def test_uncited_context_frame_does_not_count_as_grounded_evidence() -> None:
    response = {
        "answer": "con nghêu",
        "answer_status": "answered",
        "selected_candidate": {
            "video_name": "L26_V254.mp4",
            "start_sec": 10.0,
            "end_sec": 20.0,
            "evidence_frames": [
                {"frame_id": "frame-answer", "timestamp_sec": 14.0},
                {"frame_id": "actually-cited", "timestamp_sec": 16.0},
            ],
        },
        "evidence_frames": [
            {"frame_id": "actually-cited", "timestamp_sec": 16.0}
        ],
        "supporting_frame_ids": ["actually-cited"],
    }

    metrics = evaluate_response(_qrel(), response)

    assert metrics["evidence_accuracy"] == 0.0


def test_evaluator_prefers_literal_openrouter_http_request_count() -> None:
    response = {
        "usage": {
            "openrouter_calls": 4,
            "openrouter_http_requests": 7,
            "openrouter_operations": 4,
        }
    }

    metrics = evaluate_response(_qrel(), response)

    assert metrics["openrouter_calls"] == 7


def test_uncited_top_level_frame_does_not_count_as_grounded_evidence() -> None:
    response = {
        "answer": "con nghêu",
        "answer_status": "answered",
        "selected_candidate": {"video_name": "L26_V254.mp4"},
        "evidence_frames": [
            {"frame_id": "frame-answer", "timestamp_sec": 14.0},
            {"frame_id": "actually-cited", "timestamp_sec": 16.0},
        ],
        "supporting_frame_ids": ["actually-cited"],
    }

    metrics = evaluate_response(_qrel(), response)

    assert metrics["evidence_accuracy"] == 0.0


def test_point_candidate_uses_multiframe_evidence_span_for_temporal_iou() -> None:
    response = {
        "answer": "con nghêu",
        "answer_status": "answered",
        "selected_candidate": {
            "video_name": "L26_V254.mp4",
            "start_sec": 15.0,
            "end_sec": 15.0,
            "evidence_frames": [
                {"frame_id": "f1", "timestamp_sec": 12.0},
                {"frame_id": "f2", "timestamp_sec": 18.0},
            ],
        },
    }

    metrics = evaluate_response(_qrel(), response)

    assert metrics["temporal_iou"] == pytest.approx(0.6)


def test_top_k_stability_and_usage_are_aggregated() -> None:
    def stable_search(**kwargs):
        assert kwargs["top_k"] in {20, 50}
        return {
            "answer": "con nghêu",
            "selected_candidate": {"video_name": "L26_V254.mp4"},
            "candidate_answers": [{"video_name": "L26_V254.mp4"}],
            "evidence_frames": [{"frame_id": "frame-answer"}],
            "usage": {"openrouter_calls": 2},
        }

    report = evaluate_queries([_qrel()], stable_search, [20, 50])

    assert report["summary"]["answer_em"] == 1.0
    assert report["summary"]["video_recall@3"] == 1.0
    assert report["summary"]["top_k_stability"] == 1.0
    assert report["summary"]["openrouter_calls_total"] == 4
    assert report["by_top_k"]["20"]["requests"] == 1


def test_changed_answer_and_video_are_reported_as_unstable() -> None:
    def unstable_search(**kwargs):
        if kwargs["top_k"] == 20:
            return {
                "answer": "nấm",
                "selected_candidate": {"video_name": "L26_V100.mp4"},
            }
        return {
            "answer": "người phụ nữ",
            "selected_candidate": {"video_name": "L26_V200.mp4"},
        }

    report = evaluate_queries([_qrel()], unstable_search, [20, 50])

    query = report["queries"][0]
    assert query["answer_top_k_stability"] == 0.0
    assert query["video_top_k_stability"] == 0.0
    assert query["top_k_stability"] == 0.0


def test_repeated_abstention_is_stable_but_has_no_selected_moment() -> None:
    def abstaining_search(**kwargs):
        return {
            "answer": "Chưa đủ bằng chứng.",
            "answer_status": "insufficient_evidence",
            "selected_candidate": None,
            "candidate_answers": [
                {
                    "video_name": "L26_V254.mp4",
                    "start_sec": 10.0,
                    "end_sec": 20.0,
                }
            ],
        }

    report = evaluate_queries([_qrel()], abstaining_search, [20, 50])

    query = report["queries"][0]
    assert query["answer_top_k_stability"] == 1.0
    assert query["video_top_k_stability"] == 1.0
    assert query["runs"][0]["selected_video"] == ""
    assert query["runs"][0]["temporal_iou"] == 0.0


def test_load_qrels_rejects_scalar_array_fields(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "scene",
                "question": "what?",
                "acceptable_answers": ["answer"],
                "relevant_frame_ids": "frame-id",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant_frame_ids must be an array"):
        load_qrels(path)


def test_load_qrels_rejects_zero_duration_moment(tmp_path) -> None:
    path = tmp_path / "point-moment.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "scene",
                "question": "what?",
                "relevant_moments": [
                    {
                        "video_name": "L26_V254.mp4",
                        "start_sec": 15.0,
                        "end_sec": 15.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid time interval"):
        load_qrels(path)


def test_interval_iou() -> None:
    assert interval_iou((12.0, 18.0), (10.0, 20.0)) == 0.6
