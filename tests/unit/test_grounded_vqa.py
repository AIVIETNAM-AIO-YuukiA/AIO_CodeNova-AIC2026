"""Unit coverage for grounded multi-frame VQA decisions.

These tests use synthetic metadata only; the real L26_V254 qrel is exercised
by ``evaluation.vqa`` on the GPU server where the experiment artifacts live.
"""

from __future__ import annotations

import json
from dataclasses import replace

from core.types import SearchResult
from retrieval.grounded_vqa import (
    EventHit,
    GroundedVqaPipeline,
    GroundedVqaPlanner,
    VqaCandidateMoment,
    VqaConstraint,
    VqaEvent,
    VqaQueryPlan,
    VqaVerification,
    _evidence_prompt,
    _event_retrieval_variants,
    _heuristic_plan,
    _normalize_text,
    _union_variant_branches,
    _vqa_verification_worker_count,
    _verification_from_payload,
    build_candidate_moments,
)


def _result(
    frame_id: str,
    video_id: str,
    timestamp: float,
    score: float = 1.0,
    shot_id: str = "s1",
) -> SearchResult:
    return SearchResult(
        frame_id=frame_id,
        video_id=video_id,
        video_name=f"{video_id}.mp4",
        frame_path=f"frames/{video_id}/{frame_id}.jpg",
        shot_id=shot_id,
        timestamp_sec=timestamp,
        score=score,
    )


def _plan() -> VqaQueryPlan:
    return VqaQueryPlan(
        visual_prompt_en="woman handles unknown target ingredient",
        visual_prompt_vi="người phụ nữ cầm nguyên liệu chưa xác định",
        events=(
            VqaEvent(
                0,
                "woman puts four [TARGET] on plate",
                "đặt bốn [TARGET]",
                answer_bearing=True,
            ),
            VqaEvent(1, "woman holds two [TARGET]", "cầm hai [TARGET]", answer_bearing=True),
            VqaEvent(2, "woman talks to another person", "đối thoại với người khác"),
        ),
        answer_type="food",
        target_reference="X",
        constraints=("The target is a food ingredient, not a person.",),
        disallowed_entity_types=("person",),
    )


def _conflicting_final_fixture() -> tuple[
    VqaQueryPlan,
    list[VqaCandidateMoment],
    list[VqaVerification],
]:
    plan = _plan()
    first_result = _result("f1", "v1", 10.0)
    first_result_later = _result("f1-later", "v1", 11.0)
    second_result = _result("f2", "v2", 20.0)
    second_result_later = _result("f2-later", "v2", 21.0)
    first = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="v1.mp4",
        start_sec=10.0,
        end_sec=10.0,
        event_hits=(
            EventHit(0, first_result, 1, 1.0),
            EventHit(1, first_result_later, 1, 1.0),
        ),
        event_coverage=1.0,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.9,
        required_event_coverage=1.0,
        evidence_frames=[
            {"frame_id": "f1", "frame_path": "f1.jpg", "timestamp_sec": 10.0},
            {
                "frame_id": "f1-later",
                "frame_path": "f1-later.jpg",
                "timestamp_sec": 11.0,
            },
        ],
    )
    second = VqaCandidateMoment(
        candidate_id="c2",
        video_id="v2",
        video_name="v2.mp4",
        start_sec=20.0,
        end_sec=20.0,
        event_hits=(
            EventHit(0, second_result, 1, 1.0),
            EventHit(1, second_result_later, 1, 1.0),
        ),
        event_coverage=1.0,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.8,
        required_event_coverage=1.0,
        evidence_frames=[
            {"frame_id": "f2", "frame_path": "f2.jpg", "timestamp_sec": 20.0},
            {
                "frame_id": "f2-later",
                "frame_path": "f2-later.jpg",
                "timestamp_sec": 21.0,
            },
        ],
    )
    verifications = [
        VqaVerification(
            candidate_id="c1",
            verdict="supported",
            answer="con nghêu",
            entity_type="food",
            confidence=0.9,
            supporting_frame_ids=("f1", "f1-later"),
            matched_event_indices=(0, 1, 2),
            supported_constraints=plan.constraints,
            contradictions=(),
            evidence_summary="shellfish",
            event_support={0: ("f1",), 1: ("f1-later",)},
        ),
        VqaVerification(
            candidate_id="c2",
            verdict="supported",
            answer="nấm",
            entity_type="food",
            confidence=0.85,
            supporting_frame_ids=("f2", "f2-later"),
            matched_event_indices=(0, 1, 2),
            supported_constraints=plan.constraints,
            contradictions=(),
            evidence_summary="mushroom",
            event_support={0: ("f2",), 1: ("f2-later",)},
        ),
    ]
    return plan, [first, second], verifications


def test_heuristic_plan_preserves_unknown_target_and_constraints() -> None:
    plan = _heuristic_plan(
        query=(
            "Cô gái đặt 4 con X lên đĩa trắng. Sau đó cô ấy cầm 2 con X; "
            "X là nguyên liệu món ăn."
        ),
        question="Hỏi X là con gì?",
        context="",
    )

    assert plan.target_reference == "X"
    assert plan.answer_type == "food"
    assert "person" in plan.disallowed_entity_types
    assert any("[TARGET]" in event.text_vi for event in plan.events)
    assert all("nghêu" not in event.text_vi.lower() for event in plan.events)


def test_target_count_is_not_confused_with_plate_count() -> None:
    plan = _heuristic_plan(
        query=(
            "Cô gái đặt lên 1 dĩa trắng 4 con X, X là nguyên liệu món ăn. "
            "Sau đó cô ấy cầm lên 2 con X."
        ),
        question="Hỏi X là con gì?",
        context="",
    )

    target_events = [event.text_en for event in plan.events if event.answer_bearing]

    assert any("placing four [TARGET]" in text for text in target_events)
    assert all("placing one [TARGET]" not in text for text in target_events)


def test_evidence_prompt_keeps_asr_ocr_and_caption_sections() -> None:
    prompt = _evidence_prompt(
        {
            "captions": [{"frame_id": "f1", "text": "c" * 3000}],
            "ocr": [{"frame_id": "f2", "text": "o" * 3000}],
            "asr": [{"start_sec": 1.0, "end_sec": 2.0, "text": "a" * 3000}],
        }
    )

    payload = json.loads(prompt)

    assert payload["asr"]
    assert payload["ocr"]
    assert payload["captions"]
    assert len(prompt) < 5000


def test_explicit_unknown_is_masked_before_llm_translation() -> None:
    class TranslatingClient:
        last_usage = {}
        user_prompt = ""

        def complete_text(self, **kwargs):
            self.user_prompt = kwargs["user_prompt"]
            return json.dumps(
                {
                    "answer_guess": None,
                    "answer_type": "food",
                    "discriminative_cues": ["clam shell"],
                    "events": [
                        {
                            "index": 0,
                            "text_en": "woman placing four clam [TARGET] on a white plate",
                            "text_vi": "ignored",
                            "ocr_keywords": ["clam"],
                            "asr_keywords": ["nghêu"],
                            "answer_bearing": True,
                        },
                        {
                            "index": 1,
                            "text_en": "woman holding two [TARGET]",
                            "text_vi": "ignored",
                            "answer_bearing": True,
                        },
                    ],
                },
                ensure_ascii=False,
            )

    client = TranslatingClient()
    plan = GroundedVqaPlanner(client=client).plan(
        query="Cô gái đặt 4 con X lên đĩa, sau đó cầm 2 con X.",
        question="X là con gì?",
        use_llm=True,
    )

    assert plan.llm_status == "llm"
    assert plan.llm_calls == 1
    assert "[TARGET]" in client.user_prompt
    assert "con X" not in client.user_prompt
    assert all("[TARGET]" in event.text_vi for event in plan.events)
    assert "woman placing four [TARGET]" in plan.events[0].text_en
    serialized_plan = json.dumps(plan.to_dict(), ensure_ascii=False).lower()
    assert "clam" not in serialized_plan
    assert "nghêu" not in serialized_plan
    assert "nghêu" not in client.user_prompt.lower()


def test_unknown_mentioned_only_in_question_is_bound_and_translated() -> None:
    class TranslatingClient:
        last_usage = {}

        def complete_text(self, **kwargs):
            assert "[TARGET]" in kwargs["user_prompt"]
            return json.dumps(
                {
                    "answer_guess": None,
                    "answer_type": "object",
                    "events": [
                        {
                            "index": 0,
                            "text_en": "woman placing four [TARGET] on a plate",
                            "text_vi": "ignored",
                            "answer_bearing": True,
                        },
                        {
                            "index": 1,
                            "text_en": "woman holding two [TARGET]",
                            "text_vi": "ignored",
                            "answer_bearing": True,
                        },
                    ],
                },
                ensure_ascii=False,
            )

    plan = GroundedVqaPlanner(client=TranslatingClient()).plan(
        query="Cô gái đặt bốn con vật lên đĩa rồi cầm hai con vật.",
        question="X là con gì?",
        use_llm=True,
    )

    assert plan.target_reference == "X"
    assert plan.llm_status == "llm"
    assert plan.llm_calls == 1
    assert any("[TARGET]" in event.text_vi for event in plan.events)
    assert any(event.answer_bearing for event in plan.events)


def test_ordered_event_chain_beats_generic_cooking_distractor() -> None:
    plan = _plan()
    correct_events = [
        [_result("right-1", "right", 10.0)],
        [_result("right-2", "right", 20.0)],
        [_result("right-3", "right", 30.0)],
    ]
    # Generic kitchen video appears in every result list but in the wrong order.
    distractors = [
        _result("wrong-1", "wrong", 30.0),
        _result("wrong-2", "wrong", 20.0),
        _result("wrong-3", "wrong", 10.0),
    ]
    event_results = [
        [correct_events[index][0], distractors[index]] for index in range(3)
    ]
    full_results = [
        _result("wrong-global", "wrong", 15.0),
        _result("right-global", "right", 20.0),
    ]

    candidates = build_candidate_moments(plan, full_results, event_results)

    assert candidates
    assert candidates[0].video_id == "right"
    assert candidates[0].event_coverage == 1.0
    assert [hit.event_index for hit in candidates[0].event_hits] == [0, 1, 2]


def test_candidate_builder_never_chains_across_videos() -> None:
    plan = _plan()
    event_results = [
        [_result("a1", "video-a", 10.0)],
        [_result("b1", "video-b", 20.0)],
        [_result("a3", "video-a", 30.0)],
    ]

    candidates = build_candidate_moments(plan, [], event_results)

    assert candidates == []


def test_same_frame_cannot_satisfy_multiple_ordered_events() -> None:
    plan = _plan()
    repeated = _result("same-frame", "video-a", 10.0)

    candidates = build_candidate_moments(
        plan,
        [],
        [[repeated], [repeated], [repeated]],
    )

    assert candidates == []


def test_low_rank_distractors_do_not_change_best_candidate() -> None:
    plan = _plan()
    base = [
        [_result("r1", "right", 10.0)],
        [_result("r2", "right", 20.0)],
        [_result("r3", "right", 30.0)],
    ]
    before = build_candidate_moments(plan, [], base)
    # Build each event list with 49 low-ranked, single-event distractors.
    noisy = [
        results
        + [
            _result(f"noise-{event_index}-{rank}", f"noise-{event_index}-{rank}", rank)
            for rank in range(1, 50)
        ]
        for event_index, results in enumerate(base)
    ]
    after = build_candidate_moments(plan, [], noisy)

    assert before[0].video_id == "right"
    assert after[0].video_id == "right"


def test_beam_search_keeps_a_lower_early_hit_that_forms_a_complete_chain() -> None:
    plan = _plan()
    event_results = [
        [
            _result("dead-end", "right", 100.0),
            _result("chain-start", "right", 10.0),
        ],
        [_result("chain-end", "right", 20.0)],
        [],
    ]

    candidates = build_candidate_moments(
        plan,
        [],
        event_results,
        beam_width=2,
    )

    assert candidates
    assert [
        hit.result.frame_id
        for hit in candidates[0].event_hits
        if hit.event_index in {0, 1}
    ] == ["chain-start", "chain-end"]
    assert candidates[0].required_event_coverage == 1.0


def test_beam_pairs_before_pruning_a_low_rank_first_required_event(monkeypatch) -> None:
    """A valid early action must survive more than ``beam_width`` dead ends."""
    monkeypatch.setenv("VQA_CANDIDATE_VIDEO_HITS_PER_EVENT", "32")
    plan = _plan()
    dead_ends = [
        _result(f"dead-{index}", "right", 100.0 + index)
        for index in range(25)
    ]
    event_results = [
        [*dead_ends, _result("chain-start", "right", 10.0)],
        [_result("chain-end", "right", 20.0)],
        [],
    ]

    candidates = build_candidate_moments(
        plan,
        [],
        event_results,
        beam_width=5,
    )

    assert candidates
    required_ids = [
        hit.result.frame_id
        for hit in candidates[0].event_hits
        if hit.event_index in {0, 1}
    ]
    assert required_ids == ["chain-start", "chain-end"]


def test_context_quality_breaks_tie_between_complete_chains() -> None:
    plan = _plan()
    weak_context_distractors = [
        _result(f"noise-{index}", f"noise-{index}", 30.0)
        for index in range(8)
    ]
    candidates = build_candidate_moments(
        plan,
        [],
        [
            [
                _result("strong-r0", "strong", 10.0),
                _result("weak-r0", "weak", 10.0),
            ],
            [
                _result("weak-r1", "weak", 20.0),
                _result("strong-r1", "strong", 20.0),
            ],
            [
                _result("strong-context", "strong", 30.0),
                *weak_context_distractors,
                _result("weak-context", "weak", 30.0),
            ],
        ],
        candidate_count=2,
    )

    assert [candidate.video_id for candidate in candidates] == ["strong", "weak"]
    assert candidates[0].context_quality > candidates[1].context_quality


def test_context_anchor_rescue_keeps_a_complete_low_rank_candidate() -> None:
    plan = _plan()
    candidates = build_candidate_moments(
        plan,
        [
            _result("generic-full", "generic", 15.0),
            _result("target-full", "target", 15.0),
        ],
        [
            [
                _result("generic-r0", "generic", 10.0),
                _result("target-r0", "target", 10.0),
            ],
            [
                _result("generic-r1", "generic", 20.0),
                _result("target-r1", "target", 20.0),
            ],
            [
                _result("target-context", "target", 30.0),
                _result("generic-context", "generic", 30.0),
            ],
        ],
        candidate_count=2,
    )

    assert {candidate.video_id for candidate in candidates} == {"generic", "target"}
    rescued = next(candidate for candidate in candidates if candidate.video_id == "target")
    assert rescued.required_event_coverage == 1.0
    assert rescued.candidate_source == "context_anchor_rescue"
    assert "complete_required_chain" in rescued.selection_reason


def test_full_query_video_rank_counts_without_adding_a_far_frame_to_evidence() -> None:
    plan = _plan()
    candidates = build_candidate_moments(
        plan,
        [_result("far-full-hit", "right", 300.0)],
        [
            [_result("required-0", "right", 10.0)],
            [_result("required-1", "right", 20.0)],
            [],
        ],
    )

    assert candidates
    assert candidates[0].global_rank_score == 1.0
    assert candidates[0].global_hit is None


def test_variant_union_retains_a_hit_strong_in_only_one_model() -> None:
    target = _result("target", "L26_V254", 12.0)
    distractor = _result("distractor", "other", 10.0)

    merged = _union_variant_branches(
        {
            "jina-clip-v2:en:0": [distractor],
            "siglip2-so400m:en:0": [target],
        },
        top_k=10,
        max_hits_per_video=8,
    )

    assert {result.frame_id for result in merged} == {"target", "distractor"}
    target_result = next(result for result in merged if result.frame_id == "target")
    assert target_result.model_query_consensus == 0.5
    assert target_result.score == 0.9


def test_variant_union_reserves_single_branch_witness_before_video_cap() -> None:
    generic = [
        _result(f"generic-{index}", "same-video", float(index))
        for index in range(8)
    ]
    target = _result("target-action", "same-video", 90.0)
    other_branch = [
        _result(f"other-{index}", f"other-video-{index}", 10.0)
        for index in range(12)
    ]

    merged = _union_variant_branches(
        {
            "siglip:en:0": generic,
            "jina:focused:1": [*other_branch, target],
        },
        top_k=100,
        max_hits_per_video=2,
        reserve_per_branch=1,
    )

    assert {result.frame_id for result in merged if result.video_id == "same-video"} >= {
        "generic-0",
        "target-action",
    }


def test_target_safe_event_adds_short_action_variant_without_entity_guess() -> None:
    plan = _heuristic_plan(
        query="Cô gái đặt 4 con X lên đĩa trắng, sau đó cầm 2 con X.",
        question="X là con gì?",
        context="",
    )
    event = next(event for event in plan.events if event.answer_bearing)

    variants = _event_retrieval_variants(event, plan=plan)

    assert len(variants["en"]) >= 2
    assert any("four food ingredients" in value for value in variants["en"])
    serialized = json.dumps(variants, ensure_ascii=False).lower()
    assert "nghêu" not in serialized
    assert "clam" not in serialized


def test_vqa_verification_concurrency_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv("VQA_VERIFICATION_CONCURRENCY", raising=False)
    assert _vqa_verification_worker_count(5) == 1
    monkeypatch.setenv("VQA_VERIFICATION_CONCURRENCY", "3")
    assert _vqa_verification_worker_count(5) == 3
    assert _vqa_verification_worker_count(2) == 2


def test_variant_search_does_not_truncate_the_union_to_one_branch_pool() -> None:
    class BranchRetriever:
        reranker = None

        def search_variant_branches(self, query_variants, **kwargs):
            assert kwargs["top_k"] == 2
            return {
                "jina:en:0": [
                    _result("j1", "jina-1", 1.0),
                    _result("j2", "jina-2", 2.0),
                ],
                "siglip:en:0": [
                    _result("s1", "siglip-1", 3.0),
                    _result("target", "L26_V254", 4.0),
                ],
            }

    pipeline = GroundedVqaPipeline(object(), BranchRetriever())
    results, trace = pipeline._search_visual_variants(
        {"en": ["target-safe event"]},
        pool_size=2,
        enabled_models=["jina", "siglip"],
        use_reranker=False,
    )

    assert len(results) == 4
    assert any(result.frame_id == "target" for result in results)
    assert trace["per_branch_pool_size"] == 2
    assert trace["union_limit"] == 4


def test_variant_consensus_is_shared_by_neighboring_frames_in_one_video() -> None:
    merged = _union_variant_branches(
        {
            "jina:en:0": [_result("jina-frame", "L26_V254", 10.0)],
            "siglip:en:0": [_result("siglip-frame", "L26_V254", 11.0)],
        },
        top_k=10,
        max_hits_per_video=8,
    )

    assert len(merged) == 2
    assert all(result.model_query_consensus == 1.0 for result in merged)


def test_variant_consensus_does_not_cross_distant_video_moments() -> None:
    merged = _union_variant_branches(
        {
            "jina:en:0": [
                _result("early", "same-video", 10.0, shot_id="shot-a")
            ],
            "siglip:en:0": [
                _result("late", "same-video", 200.0, shot_id="shot-b")
            ],
        },
        top_k=10,
        max_hits_per_video=8,
    )

    assert len(merged) == 2
    assert all(result.model_query_consensus == 0.5 for result in merged)


def test_video_hit_cap_reserves_a_slot_for_a_later_long_shot_moment() -> None:
    same_shot = [
        _result(f"near-{index}", "video", float(index), shot_id="shot-a")
        for index in range(8)
    ]
    later_shot = _result("later", "video", 30.0, shot_id="shot-a")

    merged = _union_variant_branches(
        {"siglip:en:0": [*same_shot, later_shot]},
        top_k=8,
        max_hits_per_video=8,
    )

    assert len(merged) == 8
    assert any(result.frame_id == "later" for result in merged)


def test_variant_retrieval_honors_enabled_reranker() -> None:
    class Reranker:
        query = ""

        def rerank(self, *, query, results):
            self.query = query
            return list(reversed(results))

    class BranchRetriever:
        reranker = Reranker()

        def search_variant_branches(self, query_variants, **kwargs):
            return {
                "siglip:en:0": [
                    _result("first", "v1", 10.0),
                    _result("second", "v2", 20.0),
                ]
            }

    pipeline = GroundedVqaPipeline(object(), BranchRetriever())
    results, trace = pipeline._search_visual_variants(
        {
            "full_en": ["long full query"],
            "target_sequence_en": ["four ingredients then two ingredients"],
        },
        pool_size=500,
        enabled_models=["siglip"],
        use_reranker=True,
    )

    assert [result.frame_id for result in results] == ["second", "first"]
    assert pipeline.retriever.reranker.query == "four ingredients then two ingredients"
    assert trace["reranker_applied"] is True


def test_vietnamese_normalization_maps_d_stroke() -> None:
    assert _normalize_text("Đĩa đỏ") == "dia do"


def test_person_answer_is_rejected_for_food_target() -> None:
    plan = _plan()
    result = _result("f1", "v1", 10.0)
    candidate = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="v1.mp4",
        start_sec=8.0,
        end_sec=12.0,
        event_hits=(EventHit(0, result, 1, 1.0), EventHit(1, result, 1, 1.0)),
        event_coverage=2 / 3,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.8,
    )

    verification = _verification_from_payload(
        candidate=candidate,
        plan=plan,
        payload={
            "verdict": "supported",
            "answer": "người phụ nữ",
            "entity_type": "person",
            "confidence": 0.99,
            "supporting_frames": ["F1"],
            "matched_event_indices": [0, 1, 2],
        },
        frame_map={"F1": "f1"},
        usage={},
    )

    assert verification.verdict == "not_supported"
    assert verification.answer == "người phụ nữ"
    assert verification.contradictions


def test_supported_food_answer_requires_real_evidence_frame() -> None:
    plan = _plan()
    first = _result("f1", "v1", 10.0)
    second = _result("f2", "v1", 15.0)
    candidate = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="L26_V254.mp4",
        start_sec=8.0,
        end_sec=18.0,
        event_hits=(EventHit(0, first, 1, 1.0), EventHit(1, second, 1, 1.0)),
        event_coverage=2 / 3,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.8,
    )

    valid = _verification_from_payload(
        candidate=candidate,
        plan=plan,
        payload={
            "verdict": "supported",
            "answer": "con nghêu",
            "entity_type": "food",
            "confidence": 0.9,
            "supporting_frames": ["F1", "F2"],
            "matched_event_indices": [0, 1],
            "event_support": {"0": ["F1"], "1": ["F2"]},
            "supported_constraints": list(plan.constraints),
        },
        frame_map={"F1": "f1", "F2": "f2"},
        frame_events={"F1": {0}, "F2": {1}},
        frame_times={"F1": 10.0, "F2": 15.0},
        usage={},
    )
    fabricated = _verification_from_payload(
        candidate=candidate,
        plan=plan,
        payload={
            "verdict": "supported",
            "answer": "con nghêu",
            "entity_type": "food",
            "confidence": 0.9,
            "supporting_frames": ["not-supplied"],
            "matched_event_indices": [0, 1],
            "event_support": {
                "0": ["not-supplied"],
                "1": ["not-supplied"],
            },
            "supported_constraints": list(plan.constraints),
        },
        frame_map={"F1": "f1", "F2": "f2"},
        frame_events={"F1": {0}, "F2": {1}},
        frame_times={"F1": 10.0, "F2": 15.0},
        usage={},
    )

    assert valid.verdict == "supported"
    assert valid.supporting_frame_ids == ("f1", "f2")
    assert fabricated.verdict == "partial"
    assert fabricated.supporting_frame_ids == ()


def test_food_name_containing_man_is_not_rejected_as_person() -> None:
    plan = _plan()
    first = _result("f1", "v1", 10.0)
    second = _result("f2", "v1", 15.0)
    candidate = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="v1.mp4",
        start_sec=8.0,
        end_sec=12.0,
        event_hits=(EventHit(0, first, 1, 1.0), EventHit(1, second, 1, 1.0)),
        event_coverage=2 / 3,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.8,
    )

    verification = _verification_from_payload(
        candidate=candidate,
        plan=plan,
        payload={
            "verdict": "supported",
            "answer": "mango",
            "entity_type": "food",
            "confidence": 0.9,
            "supporting_frames": ["F1", "F2"],
            "matched_event_indices": [0, 1],
            "event_support": {"0": ["F1"], "1": ["F2"]},
            "supported_constraints": list(plan.constraints),
        },
        frame_map={"F1": "f1", "F2": "f2"},
        frame_events={"F1": {0}, "F2": {1}},
        frame_times={"F1": 10.0, "F2": 15.0},
        usage={},
    )

    assert verification.verdict == "supported"


def test_unrelated_constraint_text_cannot_satisfy_grounding() -> None:
    plan = _plan()
    result = _result("f1", "v1", 10.0)
    candidate = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="v1.mp4",
        start_sec=8.0,
        end_sec=12.0,
        event_hits=(EventHit(0, result, 1, 1.0),),
        event_coverage=1 / 3,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.7,
    )

    verification = _verification_from_payload(
        candidate=candidate,
        plan=plan,
        payload={
            "verdict": "supported",
            "answer": "con nghêu",
            "entity_type": "food",
            "confidence": 0.9,
            "supporting_frames": ["F1"],
            "matched_event_indices": [0, 1, 2],
            "supported_constraints": ["some unrelated observation"],
        },
        frame_map={"F1": "f1"},
        usage={},
    )

    assert verification.verdict == "not_supported"
    assert verification.supported_constraints == ()


def test_structured_event_support_and_constraints_are_backend_validated() -> None:
    plan = replace(
        _plan(),
        constraint_specs=(
            VqaConstraint("TARGET_ENTITY_TYPE", "entity_type", "food target", value="food"),
            VqaConstraint("E0_COUNT_4", "count", "four targets", event_index=0, value=4),
            VqaConstraint("E1_COUNT_2", "count", "two targets", event_index=1, value=2),
            VqaConstraint("ORDER_0_1", "order", "event order", value="0_1"),
        ),
    )
    first = _result("f1", "v1", 10.0)
    second = _result("f2", "v1", 15.0)
    candidate = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="L26_V254.mp4",
        start_sec=10.0,
        end_sec=15.0,
        event_hits=(EventHit(0, first, 1, 1.0), EventHit(1, second, 1, 1.0)),
        event_coverage=2 / 3,
        chain_score=1.0,
        global_rank_score=0.0,
        retrieval_score=0.9,
        required_event_coverage=1.0,
    )
    common = {
        "verdict": "supported",
        "answer": "nghêu",
        "entity_type": "food",
        "confidence": 1.0,
        "supporting_frames": ["F1", "F2"],
        "matched_event_indices": [0, 1],
        "supported_constraint_ids": [
            "TARGET_ENTITY_TYPE",
            "E0_COUNT_4",
            "E1_COUNT_2",
            "ORDER_0_1",
        ],
    }

    valid = _verification_from_payload(
        candidate=candidate,
        plan=plan,
        payload={**common, "event_support": {"0": ["F1"], "1": ["F2"]}},
        frame_map={"F1": "f1", "F2": "f2"},
        frame_events={"F1": {0}, "F2": {1}},
        frame_times={"F1": 10.0, "F2": 15.0},
        usage={},
    )
    wrong_event = _verification_from_payload(
        candidate=candidate,
        plan=plan,
        payload={**common, "event_support": {"0": ["F1"], "1": ["F1"]}},
        frame_map={"F1": "f1", "F2": "f2"},
        frame_events={"F1": {0}, "F2": {1}},
        frame_times={"F1": 10.0, "F2": 15.0},
        usage={},
    )
    missing_event_support = _verification_from_payload(
        candidate=candidate,
        plan=plan,
        payload=common,
        frame_map={"F1": "f1", "F2": "f2"},
        frame_events={"F1": {0}, "F2": {1}},
        frame_times={"F1": 10.0, "F2": 15.0},
        usage={},
    )

    assert valid.verdict == "supported"
    assert valid.effective_confidence == 1.0
    assert set(valid.validated_constraint_ids) == {
        "TARGET_ENTITY_TYPE",
        "E0_COUNT_4",
        "E1_COUNT_2",
        "ORDER_0_1",
    }
    assert wrong_event.verdict == "partial"
    assert wrong_event.valid_citation_coverage == 0.5
    assert missing_event_support.verdict == "partial"
    assert missing_event_support.valid_citation_coverage == 0.0


def test_partial_required_candidate_is_rejected_before_openrouter() -> None:
    plan = _plan()
    first = _result("f1", "v1", 10.0)
    candidate = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="v1.mp4",
        start_sec=10.0,
        end_sec=10.0,
        event_hits=(EventHit(0, first, 1, 1.0),),
        event_coverage=1 / 3,
        chain_score=1.0,
        global_rank_score=0.0,
        retrieval_score=0.5,
        evidence_frames=[
            {"frame_id": f"f{index}", "frame_path": f"f{index}.jpg"}
            for index in range(1, 5)
        ],
    )

    pipeline = object.__new__(GroundedVqaPipeline)
    verification = pipeline._verify_one(
        plan=plan,
        query="query",
        question="question",
        context="",
        candidate=candidate,
    )

    assert verification.verdict == "not_supported"
    assert verification.logical_calls == 0
    assert verification.effective_confidence == 0.0


def test_final_judge_cannot_select_candidate_with_other_candidates_frame() -> None:
    plan, candidates, verifications = _conflicting_final_fixture()

    class CrossCitingClient:
        last_usage = {}

        def complete_with_images(self, **kwargs):
            return (
                '{"status":"answered","selected_candidate_id":"c1",'
                '"answer":"con nghêu","confidence":0.9,'
                '"supporting_frames":["F3"]}'
            )

    pipeline = object.__new__(GroundedVqaPipeline)
    pipeline._vlm_client = CrossCitingClient()

    selection, _, error = pipeline._select_final_answer(
        plan=plan,
        query="query",
        question="question",
        context="",
        candidates=candidates,
        verifications=verifications,
    )

    assert selection["status"] == "insufficient_evidence"
    assert selection["candidate_id"] is None
    assert error is None


def test_final_judge_can_select_supported_conflicting_candidate() -> None:
    plan, candidates, verifications = _conflicting_final_fixture()

    class SelectingClient:
        last_usage = {}

        def complete_with_images(self, **kwargs):
            return (
                '{"status":"answered","selected_candidate_id":"c1",'
                '"answer":"con nghêu","confidence":0.91,'
                '"supporting_frames":["F1","F2"],"evidence_summary":"visible shellfish"}'
            )

    pipeline = object.__new__(GroundedVqaPipeline)
    pipeline._vlm_client = SelectingClient()

    selection, usage, error = pipeline._select_final_answer(
        plan=plan,
        query="query",
        question="question",
        context="",
        candidates=candidates,
        verifications=verifications,
    )

    assert selection["status"] == "answered"
    assert selection["candidate_id"] == "c1"
    assert selection["supporting_frame_ids"] == ("f1", "f1-later")
    assert usage["_logical_calls"] == 1
    assert error is None


def test_hidden_target_runs_final_visual_judge_even_when_candidates_agree() -> None:
    plan, candidates, verifications = _conflicting_final_fixture()
    verifications[1] = replace(verifications[1], answer="con nghêu")

    class AgreeingClient:
        last_usage = {}

        def complete_with_images(self, **kwargs):
            return (
                '{"status":"answered","selected_candidate_id":"c1",'
                '"answer":"con nghêu","confidence":0.9,'
                '"supporting_frames":["F1","F2"]}'
            )

    pipeline = object.__new__(GroundedVqaPipeline)
    pipeline._vlm_client = AgreeingClient()

    selection, usage, error = pipeline._select_final_answer(
        plan=plan,
        query="query",
        question="question",
        context="",
        candidates=candidates,
        verifications=verifications,
    )

    assert selection["status"] == "answered"
    assert selection["candidate_id"] == "c1"
    assert selection["supporting_frame_ids"] == ("f1", "f1-later")
    assert usage["_logical_calls"] == 1
    assert error is None


def test_final_openrouter_failure_abstains_without_react_fallback() -> None:
    plan, candidates, verifications = _conflicting_final_fixture()

    class FailingClient:
        last_usage = {}

        def complete_with_images(self, **kwargs):
            raise ConnectionError("upstream unavailable")

    pipeline = object.__new__(GroundedVqaPipeline)
    pipeline._vlm_client = FailingClient()

    selection, usage, error = pipeline._select_final_answer(
        plan=plan,
        query="query",
        question="question",
        context="",
        candidates=candidates,
        verifications=verifications,
    )

    assert selection["status"] == "insufficient_evidence"
    assert selection["candidate_id"] is None
    assert usage["_logical_calls"] == 1
    assert "ConnectionError" in error


def test_display_top_k_does_not_change_fixed_retrieval_pool() -> None:
    class Planner:
        def plan(self, **kwargs):
            return _plan()

    pipeline = GroundedVqaPipeline(object(), object(), planner=Planner())
    requested_pools: list[int] = []

    def no_results(plan, *, pool_size, enabled_models, use_reranker):
        requested_pools.append(pool_size)
        return [], [[] for _ in plan.events]

    pipeline._retrieve = no_results
    first = pipeline.run(query="query", question="question", top_k=20)
    second = pipeline.run(query="query", question="question", top_k=50)
    third = pipeline.run(query="query", question="question", top_k=100)

    assert requested_pools == [500, 500, 500]
    assert (
        first["answer_status"]
        == second["answer_status"]
        == third["answer_status"]
        == "no_candidates"
    )
    assert first["query_plan"] == second["query_plan"] == third["query_plan"]


def test_display_top_k_keeps_answer_candidate_and_citations_stable() -> None:
    plan = replace(_plan(), target_reference="")

    class Planner:
        def plan(self, **kwargs):
            return plan

    pipeline = GroundedVqaPipeline(object(), object(), planner=Planner())
    requested_pools: list[int] = []
    # The answer video is deliberately absent from raw full-query cards. The
    # response must still put the ordered-event candidate in display_results.
    full_results = [_result("global-distractor", "wrong", 20.0)]
    event_results = [
        [_result("right-1", "right", 10.0)],
        [_result("right-2", "right", 20.0)],
        [_result("right-3", "right", 30.0)],
    ]

    def retrieve(plan, *, pool_size, enabled_models, use_reranker):
        requested_pools.append(pool_size)
        return full_results, event_results

    def evidence(candidate, plan, *, max_frames):
        return [
            {
                "evidence_label": f"F{index}",
                "frame_id": f"{candidate.candidate_id}-e{index}",
                "video_id": candidate.video_id,
                "frame_path": f"/tmp/{candidate.candidate_id}-e{index}.jpg",
                "timestamp_sec": candidate.start_sec + index,
            }
            for index in range(1, 5)
        ]

    def verify(*, candidates, **kwargs):
        output = []
        for index, candidate in enumerate(candidates):
            output.append(
                VqaVerification(
                    candidate_id=candidate.candidate_id,
                    verdict="supported" if index == 0 else "not_supported",
                    answer="con nghêu" if index == 0 else None,
                    entity_type="food",
                    confidence=0.9 if index == 0 else 0.0,
                    supporting_frame_ids=(
                        str(candidate.evidence_frames[0]["frame_id"]),
                    )
                    if index == 0
                    else (),
                    matched_event_indices=(0, 1, 2) if index == 0 else (),
                    supported_constraints=plan.constraints if index == 0 else (),
                    contradictions=(),
                    evidence_summary="visible shellfish" if index == 0 else "",
                )
            )
        return output

    pipeline._retrieve = retrieve
    pipeline._select_evidence_frames = evidence
    pipeline._collect_text_evidence = lambda candidate: {
        "captions": [],
        "ocr": [],
        "asr": [],
    }
    pipeline._verify_candidates = verify

    first = pipeline.run(query="query", question="question", top_k=20)
    second = pipeline.run(query="query", question="question", top_k=50)
    third = pipeline.run(query="query", question="question", top_k=100)

    assert requested_pools == [500, 500, 500]
    assert first["answer"] == second["answer"] == third["answer"] == "con nghêu"
    assert first["selected_candidate"]["video_id"] == "right"
    assert second["selected_candidate"]["video_id"] == "right"
    assert third["selected_candidate"]["video_id"] == "right"
    assert (
        first["supporting_frame_ids"]
        == second["supporting_frame_ids"]
        == third["supporting_frame_ids"]
    )
    assert first["display_results"][0]["video_id"] == "right"
    assert first["display_results"][0]["candidate_source"] == "ordered_event_union"


def test_low_confidence_candidate_causes_abstention() -> None:
    plan = _plan()
    result = _result("f1", "v1", 10.0)
    candidate = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="v1.mp4",
        start_sec=8.0,
        end_sec=12.0,
        event_hits=(EventHit(0, result, 1, 1.0),),
        event_coverage=1 / 3,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.7,
    )
    verification = VqaVerification(
        candidate_id="c1",
        verdict="supported",
        answer="con nghêu",
        entity_type="food",
        confidence=0.64,
        supporting_frame_ids=("f1",),
        matched_event_indices=(0, 1),
        supported_constraints=plan.constraints,
        contradictions=(),
        evidence_summary="visible shellfish",
    )
    pipeline = object.__new__(GroundedVqaPipeline)

    selection, usage, error = pipeline._select_final_answer(
        plan=plan,
        query="query",
        question="question",
        context="",
        candidates=[candidate],
        verifications=[verification],
    )

    assert selection["status"] == "insufficient_evidence"
    assert usage == {}
    assert error is None
