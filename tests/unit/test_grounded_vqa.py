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
    VqaEvent,
    VqaQueryPlan,
    VqaVerification,
    _evidence_prompt,
    _heuristic_plan,
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
    second_result = _result("f2", "v2", 20.0)
    first = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="v1.mp4",
        start_sec=10.0,
        end_sec=10.0,
        event_hits=(EventHit(0, first_result, 1, 1.0),),
        event_coverage=1.0,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.9,
        evidence_frames=[{"frame_id": "f1", "frame_path": "f1.jpg"}],
    )
    second = VqaCandidateMoment(
        candidate_id="c2",
        video_id="v2",
        video_name="v2.mp4",
        start_sec=20.0,
        end_sec=20.0,
        event_hits=(EventHit(0, second_result, 1, 1.0),),
        event_coverage=1.0,
        chain_score=1.0,
        global_rank_score=1.0,
        retrieval_score=0.8,
        evidence_frames=[{"frame_id": "f2", "frame_path": "f2.jpg"}],
    )
    verifications = [
        VqaVerification(
            candidate_id="c1",
            verdict="supported",
            answer="con nghêu",
            entity_type="food",
            confidence=0.9,
            supporting_frame_ids=("f1",),
            matched_event_indices=(0, 1, 2),
            supported_constraints=plan.constraints,
            contradictions=(),
            evidence_summary="shellfish",
        ),
        VqaVerification(
            candidate_id="c2",
            verdict="supported",
            answer="nấm",
            entity_type="food",
            confidence=0.85,
            supporting_frame_ids=("f2",),
            matched_event_indices=(0, 1, 2),
            supported_constraints=plan.constraints,
            contradictions=(),
            evidence_summary="mushroom",
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


def test_explicit_unknown_uses_answer_neutral_guard_without_llm_call() -> None:
    class FailingClient:
        def complete_text(self, **kwargs):
            raise AssertionError("The LLM must not receive an explicit unknown target")

    plan = GroundedVqaPlanner(client=FailingClient()).plan(
        query="Cô gái đặt 4 con X lên đĩa, sau đó cầm 2 con X.",
        question="X là con gì?",
        use_llm=True,
    )

    assert plan.llm_status == "heuristic_target_guard"
    assert plan.llm_calls == 0
    assert all("[TARGET]" in event.text_vi for event in plan.events)


def test_unknown_mentioned_only_in_question_still_blocks_planner_guess() -> None:
    class FailingClient:
        def complete_text(self, **kwargs):
            raise AssertionError("The LLM must not receive an explicit unknown target")

    plan = GroundedVqaPlanner(client=FailingClient()).plan(
        query="Cô gái đặt bốn con vật lên đĩa rồi cầm hai con vật.",
        question="X là con gì?",
        use_llm=True,
    )

    assert plan.target_reference == "X"
    assert plan.llm_status == "heuristic_target_guard"
    assert plan.llm_calls == 0
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

    assert candidates
    assert all(
        len({hit.result.video_id for hit in candidate.event_hits}) == 1
        for candidate in candidates
    )
    assert all(candidate.event_coverage < 1.0 for candidate in candidates)


def test_same_frame_cannot_satisfy_multiple_ordered_events() -> None:
    plan = _plan()
    repeated = _result("same-frame", "video-a", 10.0)

    candidates = build_candidate_moments(
        plan,
        [],
        [[repeated], [repeated], [repeated]],
    )

    assert candidates
    assert all(candidate.event_coverage <= 1 / 3 for candidate in candidates)


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
    result = _result("f1", "v1", 10.0)
    candidate = VqaCandidateMoment(
        candidate_id="c1",
        video_id="v1",
        video_name="L26_V254.mp4",
        start_sec=8.0,
        end_sec=18.0,
        event_hits=(EventHit(0, result, 1, 1.0), EventHit(1, result, 1, 1.0)),
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
            "supporting_frames": ["F1"],
            "matched_event_indices": [0, 1, 2],
            "supported_constraints": list(plan.constraints),
        },
        frame_map={"F1": "f1"},
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
            "matched_event_indices": [0, 1, 2],
            "supported_constraints": list(plan.constraints),
        },
        frame_map={"F1": "f1"},
        usage={},
    )

    assert valid.verdict == "supported"
    assert valid.supporting_frame_ids == ("f1",)
    assert fabricated.verdict == "partial"
    assert fabricated.supporting_frame_ids == ()


def test_food_name_containing_man_is_not_rejected_as_person() -> None:
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
            "answer": "mango",
            "entity_type": "food",
            "confidence": 0.9,
            "supporting_frames": ["F1"],
            "matched_event_indices": [0, 1, 2],
            "supported_constraints": list(plan.constraints),
        },
        frame_map={"F1": "f1"},
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

    assert verification.verdict == "partial"
    assert verification.supported_constraints == ()


def test_final_judge_cannot_select_candidate_with_other_candidates_frame() -> None:
    plan, candidates, verifications = _conflicting_final_fixture()

    class CrossCitingClient:
        last_usage = {}

        def complete_with_images(self, **kwargs):
            return (
                '{"status":"answered","selected_candidate_id":"c1",'
                '"answer":"con nghêu","confidence":0.9,'
                '"supporting_frames":["F2"]}'
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
                '"supporting_frames":["F1"],"evidence_summary":"visible shellfish"}'
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
    assert selection["supporting_frame_ids"] == ("f1",)
    assert usage["_logical_calls"] == 1
    assert error is None


def test_agreeing_candidates_skip_extra_final_openrouter_call() -> None:
    plan, candidates, verifications = _conflicting_final_fixture()
    verifications[1] = replace(verifications[1], answer="con nghêu")
    pipeline = object.__new__(GroundedVqaPipeline)

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
    assert selection["supporting_frame_ids"] == ("f1",)
    assert usage == {}
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

    assert requested_pools == [100, 100]
    assert first["answer_status"] == second["answer_status"] == "no_candidates"
    assert first["query_plan"] == second["query_plan"]


def test_display_top_k_keeps_answer_candidate_and_citations_stable() -> None:
    plan = _plan()

    class Planner:
        def plan(self, **kwargs):
            return plan

    pipeline = GroundedVqaPipeline(object(), object(), planner=Planner())
    requested_pools: list[int] = []
    full_results = [_result("global", "right", 20.0)]
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

    assert requested_pools == [100, 100]
    assert first["answer"] == second["answer"] == "con nghêu"
    assert first["selected_candidate"]["video_id"] == "right"
    assert second["selected_candidate"]["video_id"] == "right"
    assert first["supporting_frame_ids"] == second["supporting_frame_ids"]


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
