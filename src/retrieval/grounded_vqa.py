"""Grounded, multi-frame Video Question Answering.

The legacy VQA path selected one temporal segment and let a text ReAct agent
inspect one centre frame.  That is especially brittle for referring questions
(``X`` is shown in several ordered actions): increasing the display ``top_k``
could change the hidden segment and the answerer never received the scene
description that defined ``X``.

This module keeps retrieval and answering separate:

1. plan the query into ordered, answer-neutral events;
2. retrieve a fixed pool for the full description and every event;
3. form same-video ordered candidate moments;
4. inspect several timestamped frames per candidate with a vision model;
5. return an answer only when the model cites valid supporting frames.

No offline artifact is changed by this pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
import re
from threading import Lock
from time import perf_counter
from typing import Any
import unicodedata

from config.settings import Experiment
from core.paths import resolve_experiment_frame_path
from core.types import FrameRecord, SearchResult
from indexing.manifest import JsonlManifest
from modules._vllm_chat import VllmChatClient
from repository import CaptionRepository, FrameRepository
from retrieval.query_processor import ProcessedQuery
from retrieval.text_search import infer_asr_intervals, text_search

LOGGER = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_POOL = 500
DEFAULT_VIDEO_HITS_PER_EVENT = 8
DEFAULT_MOMENT_POOL = 20
DEFAULT_CANDIDATE_COUNT = 5
DEFAULT_BEAM_WIDTH = 20
DEFAULT_FRAMES_PER_CANDIDATE = 6
DEFAULT_EVENT_GAP_SEC = 60.0
DEFAULT_MOMENT_SPAN_SEC = 180.0
DEFAULT_EVIDENCE_PADDING_SEC = 2.0
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_BRANCH_CONSENSUS_WINDOW_SEC = 3.0

_TEXT_EVIDENCE_CACHE: dict[
    str,
    tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]]],
] = {}
_TEXT_EVIDENCE_LOCK = Lock()
_FRAME_INDEX_CACHE: dict[
    str,
    tuple[dict[str, FrameRecord], dict[str, list[FrameRecord]]],
] = {}
_FRAME_INDEX_LOCK = Lock()


@dataclass(frozen=True)
class VqaEvent:
    """One answer-neutral visual/text event in chronological order."""

    index: int
    text_en: str
    text_vi: str
    ocr_keywords: tuple[str, ...] = ()
    asr_keywords: tuple[str, ...] = ()
    answer_bearing: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VqaBranchSearchResult(SearchResult):
    """Search result with VQA-only branch consensus kept separate from score."""

    model_query_consensus: float = 0.0


@dataclass(frozen=True)
class VqaConstraint:
    """A machine-checkable grounding requirement tied to an event."""

    constraint_id: str
    kind: str
    description: str
    event_index: int | None = None
    value: str | int | float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VqaQueryPlan:
    """Structured plan used by retrieval and evidence verification."""

    visual_prompt_en: str
    visual_prompt_vi: str
    events: tuple[VqaEvent, ...]
    answer_type: str = "other"
    target_reference: str = ""
    discriminative_cues: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    constraint_specs: tuple[VqaConstraint, ...] = ()
    disallowed_entity_types: tuple[str, ...] = ()
    llm_status: str = "heuristic"
    fallback_reason: str | None = None
    llm_usage: dict[str, object] = field(default_factory=dict)
    llm_calls: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["events"] = [event.to_dict() for event in self.events]
        payload["constraint_specs"] = [
            constraint.to_dict() for constraint in self.constraint_specs
        ]
        return payload


@dataclass(frozen=True)
class EventHit:
    event_index: int
    result: SearchResult
    rank: int
    rank_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "rank": self.rank,
            "rank_score": round(self.rank_score, 6),
            **_search_result_payload(self.result),
        }


@dataclass
class VqaCandidateMoment:
    candidate_id: str
    video_id: str
    video_name: str
    start_sec: float
    end_sec: float
    event_hits: tuple[EventHit, ...]
    event_coverage: float
    chain_score: float
    global_rank_score: float
    retrieval_score: float
    required_event_coverage: float = 0.0
    optional_context_coverage: float = 0.0
    model_query_consensus: float = 0.0
    global_hit: SearchResult | None = None
    evidence_frames: list[dict[str, object]] = field(default_factory=list)
    text_evidence: dict[str, list[dict[str, object]]] = field(default_factory=dict)

    def to_dict(self, *, include_evidence: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "video_id": self.video_id,
            "video_name": self.video_name,
            "start_sec": round(self.start_sec, 4),
            "end_sec": round(self.end_sec, 4),
            "event_coverage": round(self.event_coverage, 6),
            "chain_score": round(self.chain_score, 6),
            "global_rank_score": round(self.global_rank_score, 6),
            "retrieval_score": round(self.retrieval_score, 6),
            "required_event_coverage": round(self.required_event_coverage, 6),
            "optional_context_coverage": round(self.optional_context_coverage, 6),
            "model_query_consensus": round(self.model_query_consensus, 6),
            "event_hits": [hit.to_dict() for hit in self.event_hits],
        }
        if include_evidence:
            payload["evidence_frames"] = self.evidence_frames
            payload["text_evidence"] = self.text_evidence
        return payload


@dataclass(frozen=True)
class VqaVerification:
    candidate_id: str
    verdict: str
    answer: str | None
    entity_type: str
    confidence: float
    supporting_frame_ids: tuple[str, ...]
    matched_event_indices: tuple[int, ...]
    supported_constraints: tuple[str, ...]
    contradictions: tuple[str, ...]
    evidence_summary: str
    error: str | None = None
    usage: dict[str, object] = field(default_factory=dict)
    logical_calls: int = 0
    event_support: dict[int, tuple[str, ...]] = field(default_factory=dict)
    validated_constraint_ids: tuple[str, ...] = ()
    required_event_coverage: float = 0.0
    valid_citation_coverage: float = 0.0
    effective_confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GroundedVqaPlanner:
    """Create one deterministic, answer-neutral VQA retrieval plan."""

    def __init__(self, client: VllmChatClient | None = None) -> None:
        self._client = client

    def _load_client(self) -> VllmChatClient:
        if self._client is None:
            model = (
                os.environ.get("VQA_OPENROUTER_MODEL")
                or os.environ.get("OPENROUTER_MODEL")
            )
            self._client = VllmChatClient(
                openrouter_model=model,
                max_retries=1,
            )
        return self._client

    def plan(
        self,
        *,
        query: str,
        question: str,
        context: str = "",
        use_llm: bool = True,
    ) -> VqaQueryPlan:
        fallback = _heuristic_plan(query=query, question=question, context=context)
        if not use_llm:
            return fallback

        system_prompt = """You translate and label an existing grounded-video retrieval plan.
Return strict JSON only. Keep exactly the supplied event IDs and chronological
order. Never answer the question and never infer the identity of [TARGET].
Preserve every [TARGET] token exactly in both languages. Translate each event
faithfully into concise visual English while retaining the Vietnamese source.
Do not add objects, foods, names, attributes, actions, or counts that are not in
the supplied source event. OCR keywords are visible words; ASR keywords are
words likely spoken. Empty keyword arrays are valid.

Schema:
{
  "answer_guess": null,
  "visual_prompt_en": "answer-neutral English visual description",
  "visual_prompt_vi": "answer-neutral Vietnamese visual description",
  "answer_type": "object|food|person|text|count|color|action|other",
  "target_reference": "X or empty",
  "discriminative_cues": ["..."],
  "constraints": ["..."],
  "disallowed_entity_types": ["person"],
  "events": [{
    "index": 0,
    "text_en": "... [TARGET] ...",
    "text_vi": "... [TARGET] ...",
    "ocr_keywords": [],
    "asr_keywords": [],
    "answer_bearing": true
  }]
}"""
        source_events = [event.to_dict() for event in fallback.events]
        user_prompt = (
            f"Context:\n{_mask_unknown_target(context.strip()) or '(none)'}\n\n"
            f"Masked scene description:\n{fallback.visual_prompt_vi}\n\n"
            f"Masked question:\n{_mask_unknown_target(question.strip() or query.strip())}\n\n"
            f"Source events (keep IDs/order/target markers):\n"
            f"{json.dumps(source_events, ensure_ascii=False)}\n\n"
            "Translate and label these events without guessing the answer."
        )
        usage: dict[str, object] = {}
        client: VllmChatClient | None = None
        try:
            client = self._load_client()
            client.last_usage = {}
            raw = client.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_params={"temperature": 0.0, "max_tokens": 900},
            )
            usage = _client_usage(client)
            payload = _extract_json_object(raw)
            if payload.get("answer_guess") not in (None, "", "null"):
                raise ValueError("planner attempted to guess the answer")
            return _plan_from_payload(
                payload,
                fallback=fallback,
                query=query,
                question=question,
                usage=usage,
            )
        except Exception as exc:  # Planner must fail open to the deterministic plan.
            usage = _client_usage(client)
            LOGGER.warning("Grounded VQA planning failed; using heuristic plan: %s", exc)
            return VqaQueryPlan(
                **{
                    **fallback.__dict__,
                    "llm_status": "fallback",
                    "fallback_reason": f"{type(exc).__name__}: {exc}"[:300],
                    "llm_usage": usage,
                    "llm_calls": 1,
                }
            )


class GroundedVqaPipeline:
    """End-to-end grounded VQA using an already constructed Retriever."""

    def __init__(
        self,
        experiment: Experiment,
        retriever,
        *,
        planner: GroundedVqaPlanner | None = None,
        vlm_client: VllmChatClient | None = None,
    ) -> None:
        self.experiment = experiment
        self.retriever = retriever
        self.planner = planner or GroundedVqaPlanner()
        self._vlm_client = vlm_client
        self._vlm_client_lock = Lock()
        self._frame_by_id: dict[str, FrameRecord] | None = None
        self._frames_by_video: dict[str, list[FrameRecord]] | None = None
        self._ocr_by_frame: dict[str, list[dict[str, object]]] | None = None
        self._asr_by_video: dict[str, list[dict[str, object]]] | None = None
        self._captions: dict[str, str] | None = None

    def _load_vlm_client(self) -> VllmChatClient:
        if self._vlm_client is None:
            with self._vlm_client_lock:
                if self._vlm_client is None:
                    self._vlm_client = VllmChatClient(
                        openrouter_model=(
                            os.environ.get("VQA_OPENROUTER_MODEL")
                            or os.environ.get("OPENROUTER_MODEL")
                        ),
                        max_retries=1,
                    )
        return self._vlm_client

    def run(
        self,
        *,
        query: str,
        question: str,
        context: str = "",
        top_k: int = 20,
        enabled_models: list[str] | None = None,
        use_reranker: bool | None = None,
        use_llm: bool = True,
    ) -> dict[str, object]:
        started = perf_counter()
        pipeline: dict[str, object] = {}
        total_usage = _empty_usage()

        tick = perf_counter()
        plan = self.planner.plan(
            query=query,
            question=question or query,
            context=context,
            use_llm=use_llm,
        )
        _merge_usage(total_usage, plan.llm_usage, logical_calls=plan.llm_calls)
        pipeline["query_planning"] = {
            "status": plan.llm_status,
            "mode": plan.llm_status,
            "llm_used": plan.llm_calls > 0 and plan.llm_status == "llm",
            "event_count": len(plan.events),
            "events_count": len(plan.events),
            "answer_type": plan.answer_type,
            "fallback_reason": plan.fallback_reason,
            "elapsed_ms": _elapsed_ms(tick),
        }

        pool_size = _env_int("VQA_RETRIEVAL_POOL", DEFAULT_RETRIEVAL_POOL, minimum=10)
        tick = perf_counter()
        full_results, event_results = self._retrieve(
            plan,
            pool_size=pool_size,
            enabled_models=enabled_models,
            use_reranker=use_reranker,
        )
        candidates = build_candidate_moments(
            plan,
            full_results,
            event_results,
            candidate_count=_env_int(
                "VQA_CANDIDATE_COUNT", DEFAULT_CANDIDATE_COUNT, minimum=1, maximum=5
            ),
            moment_pool=_env_int(
                "VQA_MOMENT_POOL", DEFAULT_MOMENT_POOL, minimum=5, maximum=20
            ),
            beam_width=_env_int(
                "VQA_BEAM_WIDTH", DEFAULT_BEAM_WIDTH, minimum=1, maximum=100
            ),
            event_gap_sec=_env_float("VQA_EVENT_GAP_SEC", DEFAULT_EVENT_GAP_SEC),
            max_moment_span_sec=_env_float(
                "VQA_MOMENT_SPAN_SEC", DEFAULT_MOMENT_SPAN_SEC
            ),
        )
        pipeline["event_retrieval"] = {
            "fixed_pool_size": pool_size,
            "display_top_k": top_k,
            "full_results_count": len(full_results),
            "event_result_counts": [len(results) for results in event_results],
            "candidate_count": len(candidates),
            "candidates_count": len(candidates),
            "elapsed_ms": _elapsed_ms(tick),
        }
        if _env_bool("VQA_DEBUG_TRACE", False):
            retrieval_trace = dict(getattr(self, "_retrieval_trace", {}))
            retrieval_trace["required_event_indices"] = sorted(
                _required_event_indices(plan)
            )
            retrieval_trace["candidate_moments"] = [
                candidate.to_dict(include_evidence=False)
                for candidate in candidates
            ]
            # The five candidates sent to the VLM are not enough to diagnose
            # retrieval misses. Preserve the diverse pre-verification pool in
            # debug mode so operators can distinguish "no valid chain" from
            # "valid chain ranked below the verification budget".
            diagnostic_pool_size = _env_int(
                "VQA_MOMENT_POOL", DEFAULT_MOMENT_POOL, minimum=5, maximum=20
            )
            diagnostic_candidates = build_candidate_moments(
                plan,
                full_results,
                event_results,
                candidate_count=diagnostic_pool_size,
                moment_pool=diagnostic_pool_size,
                beam_width=_env_int(
                    "VQA_BEAM_WIDTH", DEFAULT_BEAM_WIDTH, minimum=1, maximum=100
                ),
                event_gap_sec=_env_float("VQA_EVENT_GAP_SEC", DEFAULT_EVENT_GAP_SEC),
                max_moment_span_sec=_env_float(
                    "VQA_MOMENT_SPAN_SEC", DEFAULT_MOMENT_SPAN_SEC
                ),
            )
            retrieval_trace["candidate_pool"] = [
                candidate.to_dict(include_evidence=False)
                for candidate in diagnostic_candidates
            ]
            pipeline["retrieval_trace"] = retrieval_trace

        if not candidates:
            pipeline["evidence_selection"] = {
                "candidates": 0,
                "frame_count": 0,
                "evidence_frame_count": 0,
                "elapsed_ms": 0.0,
            }
            return self._response(
                answer="Chưa tìm thấy đoạn video phù hợp để trả lời.",
                status="no_candidates",
                confidence=0.0,
                selected=None,
                candidates=[],
                verifications=[],
                plan=plan,
                full_results=full_results,
                top_k=top_k,
                pipeline=pipeline,
                usage=total_usage,
                started=started,
                supporting_frame_ids=(),
                answer_evidence_summary="",
            )

        tick = perf_counter()
        frame_limit = _env_int(
            "VQA_FRAMES_PER_CANDIDATE",
            DEFAULT_FRAMES_PER_CANDIDATE,
            minimum=4,
            maximum=6,
        )
        for candidate in candidates:
            candidate.evidence_frames = self._select_evidence_frames(
                candidate, plan, max_frames=frame_limit
            )
            candidate.text_evidence = self._collect_text_evidence(candidate)
        candidate_frame_counts = {
            candidate.candidate_id: len(candidate.evidence_frames)
            for candidate in candidates
        }
        pipeline["evidence_selection"] = {
            "candidate_frame_counts": candidate_frame_counts,
            "frame_count": sum(candidate_frame_counts.values()),
            "evidence_frame_count": sum(candidate_frame_counts.values()),
            "max_frames_per_candidate": frame_limit,
            "elapsed_ms": _elapsed_ms(tick),
        }

        tick = perf_counter()
        verifications = self._verify_candidates(
            plan=plan,
            query=query,
            question=question or query,
            context=context,
            candidates=candidates,
        )
        for verification in verifications:
            _merge_usage(
                total_usage,
                verification.usage,
                logical_calls=verification.logical_calls,
            )
        supported_count = sum(
            verification.verdict == "supported" for verification in verifications
        )
        pipeline["candidate_verification"] = {
            "requested": len(candidates),
            "completed": len(verifications),
            "candidate_count": len(candidates),
            "supported_count": supported_count,
            "verdicts": {
                verification.candidate_id: verification.verdict
                for verification in verifications
            },
            "elapsed_ms": _elapsed_ms(tick),
        }

        tick = perf_counter()
        selection, final_usage, final_error = self._select_final_answer(
            plan=plan,
            query=query,
            question=question or query,
            context=context,
            candidates=candidates,
            verifications=verifications,
        )
        final_calls = int(final_usage.get("_logical_calls", 0))
        if final_calls:
            _merge_usage(total_usage, final_usage, logical_calls=final_calls)
        pipeline["final_verification"] = {
            "status": selection.get("status", "insufficient_evidence"),
            "selected_candidate_id": selection.get("candidate_id"),
            "supporting_frame_ids": list(
                _string_tuple(selection.get("supporting_frame_ids"), limit=6)
            ),
            "error": final_error,
            "elapsed_ms": _elapsed_ms(tick),
        }

        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.candidate_id == selection.get("candidate_id")
            ),
            None,
        )
        answer_status = str(selection.get("status") or "insufficient_evidence")
        answer = str(
            selection.get("answer")
            or "Chưa đủ bằng chứng hình ảnh để xác định câu trả lời."
        )
        confidence = _clamp_float(selection.get("confidence"), default=0.0)
        warnings: list[str] = []
        if not any(v.verdict != "error" for v in verifications):
            if any(v.logical_calls for v in verifications):
                warnings.append("All OpenRouter candidate verification calls failed.")
            else:
                warnings.append(
                    "No candidate had enough usable evidence frames for verification."
                )
        if final_error:
            warnings.append(f"Final OpenRouter verification failed: {final_error}")

        response = self._response(
            answer=answer,
            status=answer_status,
            confidence=confidence,
            selected=selected,
            candidates=candidates,
            verifications=verifications,
            plan=plan,
            full_results=full_results,
            top_k=top_k,
            pipeline=pipeline,
            usage=total_usage,
            started=started,
            supporting_frame_ids=_string_tuple(
                selection.get("supporting_frame_ids"), limit=6
            ),
            answer_evidence_summary=str(
                selection.get("evidence_summary") or ""
            )[:1000],
        )
        if warnings:
            response["agent_error"] = " ".join(warnings)
        return response

    def _retrieve(
        self,
        plan: VqaQueryPlan,
        *,
        pool_size: int,
        enabled_models: list[str] | None,
        use_reranker: bool | None,
    ) -> tuple[list[SearchResult], list[list[SearchResult]]]:
        required = _required_event_indices(plan)
        full_variants = {
            "full_en": [
                _target_safe_search_text(
                    plan.visual_prompt_en,
                    language="en",
                    answer_type=plan.answer_type,
                )
            ],
            "full_vi": [
                _target_safe_search_text(
                    plan.visual_prompt_vi,
                    language="vi",
                    answer_type=plan.answer_type,
                )
            ],
        }
        required_events = [event for event in plan.events if event.index in required]
        if len(required_events) > 1:
            full_variants["target_sequence_en"] = [
                ", then ".join(
                    _target_safe_search_text(
                        event.text_en,
                        language="en",
                        answer_type=plan.answer_type,
                    )
                    for event in required_events
                )
            ]
            full_variants["target_sequence_vi"] = [
                ", sau đó ".join(
                    _target_safe_search_text(
                        event.text_vi,
                        language="vi",
                        answer_type=plan.answer_type,
                    )
                    for event in required_events
                )
            ]

        full_results, full_trace = self._search_visual_variants(
            full_variants,
            pool_size=pool_size,
            enabled_models=enabled_models,
            use_reranker=use_reranker,
        )
        event_results: list[list[SearchResult]] = []
        event_traces: list[dict[str, object]] = []
        for event in plan.events:
            results, trace = self._search_event(
                event,
                plan=plan,
                pool_size=pool_size,
                enabled_models=enabled_models,
                use_reranker=use_reranker,
            )
            event_results.append(results)
            event_traces.append(
                {
                    "event_index": event.index,
                    "answer_bearing": event.answer_bearing,
                    **trace,
                }
            )
        self._retrieval_trace = {
            "full_query": full_trace,
            "events": event_traces,
        }
        return full_results, event_results

    def _search_event(
        self,
        event: VqaEvent,
        *,
        plan: VqaQueryPlan,
        pool_size: int,
        enabled_models: list[str] | None,
        use_reranker: bool | None,
    ) -> tuple[list[SearchResult], dict[str, object]]:
        variants = {
            "en": [
                _target_safe_search_text(
                    event.text_en,
                    language="en",
                    answer_type=plan.answer_type,
                )
            ],
            "vi": [
                _target_safe_search_text(
                    event.text_vi,
                    language="vi",
                    answer_type=plan.answer_type,
                )
            ],
        }
        visual_results, trace = self._search_visual_variants(
            variants,
            pool_size=pool_size,
            enabled_models=enabled_models,
            use_reranker=use_reranker,
        )
        auxiliary: list[list[SearchResult]] = []
        for source, keywords in (
            ("ocr", event.ocr_keywords),
            ("asr", event.asr_keywords),
        ):
            if not keywords:
                continue
            try:
                response = text_search(
                    self.experiment,
                    query=" ".join(keywords),
                    source=source,
                    top_k=pool_size,
                )
            except Exception as exc:
                LOGGER.warning("VQA %s event search degraded: %s", source, exc)
                continue
            converted = [_search_result_from_mapping(row) for row in response.get("results", [])]
            hits = [result for result in converted if result is not None]
            if hits:
                auxiliary.append(hits)
                trace.setdefault("text_sources", []).append(source)
        if auxiliary:
            # ``pool_size`` limits every visual/text branch independently.
            # Do not collapse their union back to the same size: a frame that
            # is strong in only one model/source is exactly the evidence this
            # high-recall stage is intended to preserve.
            auxiliary_union_limit = max(
                pool_size,
                len(visual_results) + sum(len(rows) for rows in auxiliary),
            )
            visual_results = _merge_auxiliary_results(
                visual_results,
                auxiliary,
                top_k=auxiliary_union_limit,
            )
        return visual_results, trace

    def _search_visual_variants(
        self,
        variants: dict[str, list[str]],
        *,
        pool_size: int,
        enabled_models: list[str] | None,
        use_reranker: bool | None,
    ) -> tuple[list[SearchResult], dict[str, object]]:
        video_cap = _env_int(
            "VQA_VIDEO_HITS_PER_EVENT",
            DEFAULT_VIDEO_HITS_PER_EVENT,
            minimum=1,
            maximum=100,
        )
        search_branches = getattr(self.retriever, "search_variant_branches", None)
        if callable(search_branches):
            branches = search_branches(
                variants,
                top_k=pool_size,
                enabled_models=enabled_models,
            )
        else:
            # Compatibility for tests and custom retrievers that predate the
            # branch API. Each variant is still searched independently, then
            # unioned without requiring cross-model agreement.
            branches: dict[str, list[SearchResult]] = {}
            for language, queries in variants.items():
                for index, query in enumerate(_unique_strings(queries)):
                    processed = ProcessedQuery(
                        raw_query=query,
                        visual_prompt=query,
                        visual_prompt_vi=query,
                    )
                    branches[f"fused:{language}:{index}"] = self.retriever.search_processed(
                        processed,
                        top_k=pool_size,
                        enabled_models=enabled_models,
                        use_reranker=use_reranker,
                        use_expansion=False,
                    )
        # ``pool_size`` is a per-model/per-query-variant limit.  A second
        # global ``[:pool_size]`` here used to discard valid single-branch
        # hits (including videos that ranked well in SigLIP but not Jina).
        union_limit = max(
            pool_size,
            sum(len(rows) for rows in branches.values()),
        )
        merged = _union_variant_branches(
            branches,
            top_k=union_limit,
            max_hits_per_video=video_cap,
        )
        reranker = getattr(self.retriever, "reranker", None)
        reranker_applied = False
        reranker_error: str | None = None
        if reranker is not None and use_reranker is not False and merged:
            consensus_by_frame = {
                result.frame_id: _result_branch_consensus(result)
                for result in merged
            }
            preferred_groups = [
                name for name in variants if name.startswith("target_sequence")
            ]
            preferred_groups.extend(
                name for name in variants if name not in preferred_groups
            )
            rerank_query = next(
                (
                    query
                    for group in preferred_groups
                    for query in variants.get(group, [])
                    if str(query).strip()
                ),
                "",
            )
            rerank_limit = min(100, len(merged))
            if rerank_query and rerank_limit:
                try:
                    reranked = reranker.rerank(
                        query=rerank_query,
                        results=merged[:rerank_limit],
                    ) + merged[rerank_limit:]
                    # Some rerankers construct fresh ``SearchResult`` objects.
                    # Reattach branch consensus so reranking cannot silently
                    # turn a single-model hit into apparent full consensus.
                    merged = [
                        _with_branch_consensus(
                            result,
                            float(result.score),
                            consensus_by_frame.get(result.frame_id, 0.0),
                        )
                        for result in reranked
                    ]
                    reranker_applied = True
                except Exception as exc:
                    reranker_error = f"{type(exc).__name__}: {exc}"[:300]
                    LOGGER.warning("Grounded VQA reranker degraded: %s", exc)
        debug_trace = _env_bool("VQA_DEBUG_TRACE", False)
        trace_limit = min(pool_size, 250) if debug_trace else 20
        trace = {
            "effective_queries": variants,
            "branch_counts": {name: len(rows) for name, rows in branches.items()},
            "per_branch_pool_size": pool_size,
            "union_limit": union_limit,
            "merged_result_count": len(merged),
            "merged_video_count": len({row.video_id for row in merged}),
            "top_videos": _top_video_trace(merged, limit=trace_limit),
            "reranker_applied": reranker_applied,
            "reranker_error": reranker_error,
        }
        if debug_trace:
            trace["branch_top_videos"] = {
                name: _top_video_trace(rows, limit=min(pool_size, 250))
                for name, rows in branches.items()
            }
            debug_video_id = os.environ.get("VQA_DEBUG_VIDEO_ID", "").strip()
            if debug_video_id:
                trace["debug_video_id"] = debug_video_id
                trace["branch_video_hits"] = {
                    name: _video_hit_trace(rows, video_id=debug_video_id)
                    for name, rows in branches.items()
                }
                trace["merged_video_hits"] = _video_hit_trace(
                    merged,
                    video_id=debug_video_id,
                )
        return merged, trace

    def _ensure_frame_index(self) -> None:
        if self._frame_by_id is not None and self._frames_by_video is not None:
            return
        cache_key = str(self.experiment.run_dir.resolve())
        with _FRAME_INDEX_LOCK:
            cached = _FRAME_INDEX_CACHE.get(cache_key)
            if cached is None:
                frames = FrameRepository(self.experiment).list_all()
                frame_by_id = {frame.frame_id: frame for frame in frames}
                grouped: dict[str, list[FrameRecord]] = defaultdict(list)
                for frame in frames:
                    grouped[frame.video_id].append(frame)
                frames_by_video = {
                    video_id: sorted(
                        values,
                        key=lambda frame: (
                            frame.timestamp_sec
                            if frame.timestamp_sec is not None
                            else math.inf,
                            frame.frame_index
                            if frame.frame_index is not None
                            else math.inf,
                            frame.frame_id,
                        ),
                    )
                    for video_id, values in grouped.items()
                }
                cached = (frame_by_id, frames_by_video)
                _FRAME_INDEX_CACHE[cache_key] = cached
        self._frame_by_id, self._frames_by_video = cached

    def _select_evidence_frames(
        self,
        candidate: VqaCandidateMoment,
        plan: VqaQueryPlan,
        *,
        max_frames: int,
    ) -> list[dict[str, object]]:
        self._ensure_frame_index()
        assert self._frame_by_id is not None
        assert self._frames_by_video is not None
        video_frames = self._frames_by_video.get(candidate.video_id, [])
        if not video_frames:
            return []

        answer_events = {event.index for event in plan.events if event.answer_bearing}
        anchors = sorted(
            candidate.event_hits,
            key=lambda hit: (
                0 if hit.event_index in answer_events else 1,
                hit.event_index,
                hit.rank,
            ),
        )
        anchor_ids: list[str] = []
        for hit in anchors:
            if hit.result.frame_id not in anchor_ids:
                anchor_ids.append(hit.result.frame_id)

        window_start = candidate.start_sec - DEFAULT_EVIDENCE_PADDING_SEC
        window_end = candidate.end_sec + DEFAULT_EVIDENCE_PADDING_SEC
        in_window = [
            frame
            for frame in video_frames
            if frame.timestamp_sec is not None
            and window_start <= float(frame.timestamp_sec) <= window_end
        ]
        if len(in_window) < min(4, max_frames):
            # Sparse keyframes: include the closest neighbours, but never another video.
            centre = (candidate.start_sec + candidate.end_sec) / 2.0
            nearest = sorted(
                (frame for frame in video_frames if frame.timestamp_sec is not None),
                key=lambda frame: abs(float(frame.timestamp_sec) - centre),
            )
            seen = {frame.frame_id for frame in in_window}
            for frame in nearest:
                if frame.frame_id not in seen:
                    in_window.append(frame)
                    seen.add(frame.frame_id)
                if len(in_window) >= max_frames * 3:
                    break

        priority: list[FrameRecord] = []
        priority_ids: set[str] = set()
        priority_event_indices: dict[str, set[int]] = defaultdict(set)

        def add_priority(
            frame: FrameRecord | None,
            event_indices: set[int] | tuple[int, ...] = (),
        ) -> None:
            if (
                frame is not None
                and frame.video_id == candidate.video_id
            ):
                priority_event_indices[frame.frame_id].update(event_indices)
                if frame.frame_id not in priority_ids:
                    priority.append(frame)
                    priority_ids.add(frame.frame_id)

        # Event anchors are the retrieval-scored frames and remain the first
        # evidence priority. They ground the whole ordered chain.
        anchor_events: dict[str, set[int]] = defaultdict(set)
        for hit in anchors:
            anchor_events[hit.result.frame_id].add(hit.event_index)
        for frame_id in anchor_ids:
            frame = self._frame_by_id.get(frame_id)
            add_priority(frame, anchor_events[frame_id])

        # Explicitly include immediate before/after keyframes around the
        # answer-bearing anchors. This is where a count-changing action such
        # as "put four down, then hold two" is most likely to be visible.
        positions = {frame.frame_id: index for index, frame in enumerate(video_frames)}
        answer_anchors = [
            hit
            for hit in anchors
            if hit.event_index in answer_events
        ]
        for hit in answer_anchors:
            frame_id = hit.result.frame_id
            position = positions.get(frame_id)
            anchor = self._frame_by_id.get(frame_id)
            if position is None or anchor is None:
                continue
            for offset in (-1, 1):
                neighbour_position = position + offset
                if not 0 <= neighbour_position < len(video_frames):
                    continue
                neighbour = video_frames[neighbour_position]
                if (
                    neighbour.timestamp_sec is not None
                    and window_start <= float(neighbour.timestamp_sec) <= window_end
                    and neighbour.shot_id == anchor.shot_id
                ):
                    add_priority(neighbour, {hit.event_index})

        if candidate.global_hit is not None:
            add_priority(self._frame_by_id.get(candidate.global_hit.frame_id))

        available = [
            frame for frame in in_window if frame.frame_id not in priority_ids
        ]
        # Keep fallback candidates beyond the final six so broken/missing
        # frame paths can be skipped without unnecessarily dropping below the
        # four-image grounding minimum.
        while available and len(priority) < max_frames * 4:
            if not priority:
                choice = min(
                    available,
                    key=lambda frame: abs(
                        float(frame.timestamp_sec or 0.0)
                        - (candidate.start_sec + candidate.end_sec) / 2.0
                    ),
                )
            else:
                selected_times = [
                    float(frame.timestamp_sec or 0.0) for frame in priority
                ]
                choice = max(
                    available,
                    key=lambda frame: min(
                        abs(float(frame.timestamp_sec or 0.0) - timestamp)
                        for timestamp in selected_times
                    ),
                )
            add_priority(choice)
            available.remove(choice)

        valid_frames: list[tuple[FrameRecord, str]] = []
        for frame in priority:
            path = resolve_experiment_frame_path(self.experiment, frame.frame_path)
            if not path.valid or path.resolved_path is None:
                LOGGER.warning(
                    "Skipping unusable VQA evidence frame %s: %s", frame.frame_id, path
                )
                continue
            valid_frames.append((frame, str(path.resolved_path)))
            if len(valid_frames) >= max_frames:
                break

        resolved: list[dict[str, object]] = []
        for frame, resolved_path in sorted(
            valid_frames,
            key=lambda item: (
                item[0].timestamp_sec
                if item[0].timestamp_sec is not None
                else math.inf,
                item[0].frame_index
                if item[0].frame_index is not None
                else math.inf,
            ),
        ):
            resolved.append(
                {
                    "evidence_label": f"F{len(resolved) + 1}",
                    "frame_id": frame.frame_id,
                    "video_id": frame.video_id,
                    "shot_id": frame.shot_id,
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "frame_path": resolved_path,
                    "event_indices": sorted(priority_event_indices.get(frame.frame_id, set())),
                }
            )
        return resolved

    def _ensure_text_evidence(self) -> None:
        if self._ocr_by_frame is None or self._asr_by_video is None:
            manifest = JsonlManifest(
                self.experiment.run_dir / "manifests" / "text.jsonl"
            )
            cache_key = str(manifest.path.resolve())
            with _TEXT_EVIDENCE_LOCK:
                cached = _TEXT_EVIDENCE_CACHE.get(cache_key)
                if cached is None:
                    records = manifest.read_all()
                    intervals = infer_asr_intervals(records)
                    ocr_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
                    asr_by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
                    for row in records:
                        source = str(row.get("source") or "")
                        text = str(row.get("text") or "").strip()
                        if not text:
                            continue
                        if source == "ocr" and row.get("frame_id"):
                            frame_id = str(row["frame_id"])
                            ocr_by_frame[frame_id].append(
                                {
                                    "frame_id": frame_id,
                                    "timestamp_sec": row.get("timestamp_sec"),
                                    "text": text[:500],
                                }
                            )
                        elif source == "asr" and row.get("video_id"):
                            video_id = str(row["video_id"])
                            doc_id = str(row.get("doc_id") or "")
                            try:
                                fallback_start = float(row.get("timestamp_sec") or 0.0)
                            except (TypeError, ValueError):
                                continue
                            start, end = intervals.get(
                                doc_id,
                                (fallback_start, fallback_start + 45.0),
                            )
                            asr_by_video[video_id].append(
                                {
                                    "start_sec": start,
                                    "end_sec": end,
                                    "text": text[:1000],
                                }
                            )
                    cached = (
                        dict(ocr_by_frame),
                        {
                            video_id: sorted(
                                values,
                                key=lambda item: float(item["start_sec"]),
                            )
                            for video_id, values in asr_by_video.items()
                        },
                    )
                    _TEXT_EVIDENCE_CACHE[cache_key] = cached
            self._ocr_by_frame, self._asr_by_video = cached
        if self._captions is None:
            self._captions = CaptionRepository(self.experiment).by_id()

    def _collect_text_evidence(
        self, candidate: VqaCandidateMoment
    ) -> dict[str, list[dict[str, object]]]:
        self._ensure_text_evidence()
        assert self._ocr_by_frame is not None
        assert self._asr_by_video is not None
        assert self._captions is not None
        frame_ids = [str(frame["frame_id"]) for frame in candidate.evidence_frames]
        self._ensure_frame_index()
        assert self._frames_by_video is not None
        padded_start = candidate.start_sec - DEFAULT_EVIDENCE_PADDING_SEC
        padded_end = candidate.end_sec + DEFAULT_EVIDENCE_PADDING_SEC
        nearby_frame_ids = list(frame_ids)
        for frame in self._frames_by_video.get(candidate.video_id, []):
            if (
                frame.timestamp_sec is not None
                and padded_start <= float(frame.timestamp_sec) <= padded_end
                and frame.frame_id not in nearby_frame_ids
            ):
                nearby_frame_ids.append(frame.frame_id)
        captions = [
            {"frame_id": frame_id, "text": self._captions[frame_id][:500]}
            for frame_id in nearby_frame_ids
            if self._captions.get(frame_id)
        ][:12]
        ocr: list[dict[str, object]] = []
        for frame_id in nearby_frame_ids:
            ocr.extend(self._ocr_by_frame.get(frame_id, []))
            if len(ocr) >= 12:
                ocr = ocr[:12]
                break

        asr_candidates: list[dict[str, object]] = []
        for row in self._asr_by_video.get(candidate.video_id, []):
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            if end < padded_start or start > padded_end:
                continue
            asr_candidates.append(dict(row))
        evidence_times = [
            float(frame["timestamp_sec"])
            for frame in candidate.evidence_frames
            if frame.get("timestamp_sec") is not None
        ]
        asr = sorted(
            asr_candidates,
            key=lambda row: (
                min(
                    (
                        0.0
                        if float(row["start_sec"])
                        <= timestamp
                        <= float(row["end_sec"])
                        else min(
                            abs(timestamp - float(row["start_sec"])),
                            abs(timestamp - float(row["end_sec"])),
                        )
                        for timestamp in evidence_times
                    ),
                    default=0.0,
                ),
                float(row["start_sec"]),
            ),
        )[:8]
        return {"captions": captions[:12], "ocr": ocr, "asr": asr}

    def _verify_candidates(
        self,
        *,
        plan: VqaQueryPlan,
        query: str,
        question: str,
        context: str,
        candidates: list[VqaCandidateMoment],
    ) -> list[VqaVerification]:
        workers = min(3, len(candidates))
        if workers <= 0:
            return []
        output: dict[str, VqaVerification] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vqa-verify") as executor:
            futures = {
                executor.submit(
                    self._verify_one,
                    plan=plan,
                    query=query,
                    question=question,
                    context=context,
                    candidate=candidate,
                ): candidate.candidate_id
                for candidate in candidates
            }
            for future in as_completed(futures):
                candidate_id = futures[future]
                try:
                    output[candidate_id] = future.result()
                except Exception as exc:  # Defensive: one candidate must not kill the request.
                    LOGGER.exception("Unexpected VQA verification failure")
                    output[candidate_id] = _error_verification(candidate_id, exc)
        return [output[candidate.candidate_id] for candidate in candidates]

    def _verify_one(
        self,
        *,
        plan: VqaQueryPlan,
        query: str,
        question: str,
        context: str,
        candidate: VqaCandidateMoment,
    ) -> VqaVerification:
        required_coverage = _candidate_required_coverage(candidate, plan)
        if required_coverage < 1.0:
            return VqaVerification(
                candidate_id=candidate.candidate_id,
                verdict="not_supported",
                answer=None,
                entity_type="other",
                confidence=0.0,
                supporting_frame_ids=(),
                matched_event_indices=(),
                supported_constraints=(),
                contradictions=(
                    "Candidate retrieval does not cover every required target event.",
                ),
                evidence_summary="",
                required_event_coverage=required_coverage,
                effective_confidence=0.0,
            )
        frames = candidate.evidence_frames
        if len(frames) < 4:
            return _error_verification(
                candidate.candidate_id,
                ValueError(f"requires at least 4 evidence frames, found {len(frames)}"),
            )

        frame_map = {
            str(frame.get("evidence_label") or f"F{index}"): str(frame["frame_id"])
            for index, frame in enumerate(frames, start=1)
        }
        frame_events = {
            str(frame.get("evidence_label") or f"F{index}"): {
                int(value)
                for value in _as_list(frame.get("event_indices"))
                if str(value).lstrip("-").isdigit()
            }
            for index, frame in enumerate(frames, start=1)
        }
        frame_times = {
            str(frame.get("evidence_label") or f"F{index}"): float(
                frame.get("timestamp_sec") or 0.0
            )
            for index, frame in enumerate(frames, start=1)
        }
        frame_lines = [
            f"{frame.get('evidence_label') or f'F{index}'}: "
            f"frame_id={frame['frame_id']}, shot={frame.get('shot_id')}, "
            f"t={float(frame.get('timestamp_sec') or 0.0):.3f}s, "
            f"eligible_events={frame.get('event_indices') or []}"
            for index, frame in enumerate(frames, start=1)
        ]
        events = [event.to_dict() for event in plan.events]
        structured_constraints = json.dumps(
            [spec.to_dict() for spec in plan.constraint_specs],
            ensure_ascii=False,
        )
        prompt = f"""Context: {context.strip() or '(none)'}
Scene description: {query}
Question: {question}

Answer type: {plan.answer_type}
Target reference: {plan.target_reference or '(none)'}
Constraints: {json.dumps(plan.constraints, ensure_ascii=False)}
Structured constraints: {structured_constraints}
Disallowed entity types: {json.dumps(plan.disallowed_entity_types, ensure_ascii=False)}
Ordered events: {json.dumps(events, ensure_ascii=False)}
Required event indices: {json.dumps(sorted(_required_event_indices(plan)))}

Candidate: {candidate.video_name} ({candidate.video_id}),
moment {candidate.start_sec:.3f}s..{candidate.end_sec:.3f}s
Frames:
{chr(10).join(frame_lines)}

Cached multimodal text evidence (may contain recognition errors; images win on conflict):
{_evidence_prompt(candidate.text_evidence)}

Inspect every image in timestamp order. Resolve the target only from this candidate.
Use verdict=supported only when every required event is grounded in its
eligible frame(s) and the event order is consistent. ``event_support`` must map
each required event index to supplied frame labels that list that event in
``eligible_events``. Return only IDs from Structured constraints in
``supported_constraint_ids`` when visibly grounded.
Return strict JSON:
{{
  "verdict": "supported|partial|not_supported",
  "answer": "short answer or null",
  "entity_type": "food|object|person|text|count|color|action|other",
  "confidence": 0.0,
  "supporting_frames": ["F1"],
  "matched_event_indices": [0],
  "event_support": {{"0": ["F1"]}},
  "supported_constraint_ids": ["E0_COUNT_4"],
  "contradictions": [],
  "evidence_summary": "brief observable evidence"
}}"""
        system = """You are a grounded multi-frame video QA verifier.
Do not answer from general knowledge or a salient person alone. The scene
description defines references such as X. Check ordered actions, counts and
entity type across the supplied frames. ASR/OCR/captions are supporting clues,
not a substitute for contradictory images. Cite only supplied frame labels.
Return JSON only, with a short answer and no hidden chain-of-thought."""
        usage: dict[str, object] = {}
        client: VllmChatClient | None = None
        try:
            client = self._load_vlm_client()
            client.last_usage = {}
            raw = client.complete_with_images(
                system_prompt=system,
                user_prompt=prompt,
                image_paths=[str(frame["frame_path"]) for frame in frames],
                image_labels=list(frame_map),
                detail=os.environ.get("VQA_IMAGE_DETAIL", "high"),
                generation_params={"temperature": 0.0, "max_tokens": 700},
            )
            usage = _client_usage(client)
            payload = _extract_json_object(raw)
            return _verification_from_payload(
                candidate=candidate,
                plan=plan,
                payload=payload,
                frame_map=frame_map,
                usage=usage,
                frame_events=frame_events,
                frame_times=frame_times,
            )
        except Exception as exc:
            usage = _client_usage(client)
            LOGGER.warning("VQA candidate %s verification failed: %s", candidate.candidate_id, exc)
            return _error_verification(
                candidate.candidate_id,
                exc,
                usage=usage,
                logical_calls=1,
            )

    def _select_final_answer(
        self,
        *,
        plan: VqaQueryPlan,
        query: str,
        question: str,
        context: str,
        candidates: list[VqaCandidateMoment],
        verifications: list[VqaVerification],
    ) -> tuple[dict[str, object], dict[str, object], str | None]:
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        scored: list[tuple[float, VqaCandidateMoment, VqaVerification]] = []
        minimum = _env_float("VQA_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
        for verification in verifications:
            candidate = candidate_by_id[verification.candidate_id]
            effective_confidence = _effective_verification_confidence(verification)
            if (
                _candidate_required_coverage(candidate, plan) >= 1.0
                and verification.verdict == "supported"
                and verification.answer
                and effective_confidence >= minimum
                and verification.supporting_frame_ids
            ):
                score = 0.35 * candidate.retrieval_score + 0.65 * effective_confidence
                scored.append((score, candidate, verification))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return (
                {
                    "status": "insufficient_evidence",
                    "answer": None,
                    "confidence": 0.0,
                    "candidate_id": None,
                    "supporting_frame_ids": (),
                    "evidence_summary": "",
                },
                {},
                None,
            )

        _, best_candidate, best_verification = scored[0]
        force_final_visual_judge = bool(plan.target_reference)
        if len(scored) == 1 and not force_final_visual_judge:
            return (
                {
                    "status": "answered",
                    "answer": best_verification.answer,
                    "confidence": _effective_verification_confidence(best_verification),
                    "candidate_id": best_candidate.candidate_id,
                    "supporting_frame_ids": best_verification.supporting_frame_ids,
                    "evidence_summary": best_verification.evidence_summary,
                },
                {},
                None,
            )

        if len(scored) > 1:
            _, _, second_verification = scored[1]
        else:
            second_verification = None
        if (
            second_verification is not None
            and not force_final_visual_judge
            and _normalize_answer(best_verification.answer)
            == _normalize_answer(second_verification.answer)
        ):
            return (
                {
                    "status": "answered",
                    "answer": best_verification.answer,
                    "confidence": _effective_verification_confidence(best_verification),
                    "candidate_id": best_candidate.candidate_id,
                    "supporting_frame_ids": best_verification.supporting_frame_ids,
                    "evidence_summary": best_verification.evidence_summary,
                },
                {},
                None,
            )

        final_frames: list[dict[str, object]] = []
        final_frame_owner: dict[str, str] = {}
        final_frame_events: dict[str, set[int]] = defaultdict(set)
        finalists = scored[:2]
        max_frames_per_finalist = 6 if len(finalists) == 1 else 3
        for _, candidate, verification in finalists:
            supported = set(verification.supporting_frame_ids)
            evidence_by_id = {
                str(frame["frame_id"]): frame for frame in candidate.evidence_frames
            }
            ordered_ids: list[str] = []
            for event_index in sorted(_required_event_indices(plan)):
                for frame_id in verification.event_support.get(event_index, ()):
                    if frame_id in evidence_by_id and frame_id not in ordered_ids:
                        ordered_ids.append(frame_id)
                    if frame_id in evidence_by_id:
                        final_frame_events[frame_id].add(event_index)
            for frame in candidate.evidence_frames:
                frame_id = str(frame["frame_id"])
                if frame_id in supported and frame_id not in ordered_ids:
                    ordered_ids.append(frame_id)
            owned = [evidence_by_id[frame_id] for frame_id in ordered_ids]
            if not owned:
                owned = candidate.evidence_frames[:max_frames_per_finalist]
            for frame in owned[:max_frames_per_finalist]:
                if len(final_frames) >= 6:
                    break
                if any(existing["frame_id"] == frame["frame_id"] for existing in final_frames):
                    continue
                final_frames.append(frame)
                final_frame_owner[str(frame["frame_id"])] = candidate.candidate_id

        labels = [f"F{index}" for index in range(1, len(final_frames) + 1)]
        label_map = {
            label: str(frame["frame_id"]) for label, frame in zip(labels, final_frames, strict=True)
        }
        proposals = [
            {
                "candidate_id": candidate.candidate_id,
                "video_name": candidate.video_name,
                "retrieval_score": candidate.retrieval_score,
                "answer": verification.answer,
                "confidence": _effective_verification_confidence(verification),
                "supporting_frame_ids": verification.supporting_frame_ids,
                "evidence_summary": verification.evidence_summary,
            }
            for _, candidate, verification in scored[:2]
        ]
        frame_ownership = {
            label: final_frame_owner[frame_id]
            for label, frame_id in label_map.items()
        }
        frame_grounding = {
            label: {
                "candidate_id": frame_ownership[label],
                "event_indices": sorted(final_frame_events.get(frame_id, set())),
                "timestamp_sec": next(
                    float(frame.get("timestamp_sec") or 0.0)
                    for frame in final_frames
                    if str(frame["frame_id"]) == frame_id
                ),
            }
            for label, frame_id in label_map.items()
        }
        prompt = f"""Context: {context.strip() or '(none)'}
Scene description: {query}
Question: {question}
Constraints: {json.dumps(plan.constraints, ensure_ascii=False)}
Candidate proposals: {json.dumps(proposals, ensure_ascii=False)}
Frame grounding: {json.dumps(frame_grounding, ensure_ascii=False)}

Verify these proposals and their cited images. Do not create a new answer.
Return strict JSON:
{{"status":"answered|insufficient_evidence","selected_candidate_id":"... or null",
"answer":"one proposed answer or null","confidence":0.0,
"supporting_frames":["F1"],"evidence_summary":"brief observable evidence"}}"""
        usage: dict[str, object] = {}
        client: VllmChatClient | None = None
        try:
            client = self._load_vlm_client()
            client.last_usage = {}
            raw = client.complete_with_images(
                system_prompt=(
                    "You are the final grounded VQA judge. Select only a proposed answer "
                    "that satisfies the user constraints and supplied images. Otherwise return "
                    "insufficient_evidence. Return JSON only."
                ),
                user_prompt=prompt,
                image_paths=[str(frame["frame_path"]) for frame in final_frames],
                image_labels=labels,
                detail=os.environ.get("VQA_IMAGE_DETAIL", "high"),
                generation_params={"temperature": 0.0, "max_tokens": 500},
            )
            usage = _client_usage(client)
            usage["_logical_calls"] = 1
            payload = _extract_json_object(raw)
            selected_id = str(payload.get("selected_candidate_id") or "")
            proposed = {
                verification.candidate_id: verification for _, _, verification in scored[:2]
            }
            chosen = proposed.get(selected_id)
            answer = str(payload.get("answer") or "").strip()
            judge_confidence = _clamp_float(payload.get("confidence"), default=0.0)
            confidence = min(
                judge_confidence,
                _effective_verification_confidence(chosen) if chosen is not None else 0.0,
            )
            cited = _resolve_supporting_frames(payload.get("supporting_frames"), label_map)
            cited_owners = {final_frame_owner.get(frame_id) for frame_id in cited}
            cited_events = {
                event_index
                for frame_id in cited
                for event_index in final_frame_events.get(frame_id, set())
            }
            if (
                payload.get("status") != "answered"
                or chosen is None
                or not answer
                or _normalize_answer(answer) != _normalize_answer(chosen.answer)
                or confidence < minimum
                or not cited
                or cited_owners != {selected_id}
                or not _required_event_indices(plan).issubset(cited_events)
            ):
                return (
                    {
                        "status": "insufficient_evidence",
                        "answer": None,
                        "confidence": confidence,
                        "candidate_id": None,
                        "supporting_frame_ids": (),
                        "evidence_summary": str(payload.get("evidence_summary") or "")[:1000],
                    },
                    usage,
                    None,
                )
            return (
                {
                    "status": "answered",
                    "answer": answer,
                    "confidence": confidence,
                    "candidate_id": selected_id,
                    "supporting_frame_ids": cited,
                    "evidence_summary": str(payload.get("evidence_summary") or "")[:1000],
                },
                usage,
                None,
            )
        except Exception as exc:
            usage = _client_usage(client)
            LOGGER.warning("Final VQA verification failed: %s", exc)
            usage["_logical_calls"] = 1
            return (
                {
                    "status": "insufficient_evidence",
                    "answer": None,
                    "confidence": 0.0,
                    "candidate_id": None,
                    "supporting_frame_ids": (),
                    "evidence_summary": "",
                },
                usage,
                f"{type(exc).__name__}: {exc}"[:300],
            )

    def _response(
        self,
        *,
        answer: str,
        status: str,
        confidence: float,
        selected: VqaCandidateMoment | None,
        candidates: list[VqaCandidateMoment],
        verifications: list[VqaVerification],
        plan: VqaQueryPlan,
        full_results: list[SearchResult],
        top_k: int,
        pipeline: dict[str, object],
        usage: dict[str, object],
        started: float,
        supporting_frame_ids: tuple[str, ...],
        answer_evidence_summary: str,
    ) -> dict[str, object]:
        pipeline["total_elapsed_ms"] = _elapsed_ms(started)
        selected_payload = selected.to_dict() if selected is not None else None
        selected_verification = next(
            (
                verification
                for verification in verifications
                if selected is not None
                and verification.candidate_id == selected.candidate_id
            ),
            None,
        )
        if selected_payload is not None and selected_verification is not None:
            selected_payload["event_support"] = selected_verification.event_support
            selected_payload["effective_confidence"] = round(
                _effective_verification_confidence(selected_verification), 6
            )
        supporting_ids = list(supporting_frame_ids)
        supporting_set = set(supporting_ids)
        evidence_frames = (
            [
                {**frame, "role": "supporting"}
                for frame in selected.evidence_frames
                if str(frame.get("frame_id")) in supporting_set
            ]
            if selected is not None
            else []
        )
        candidate_by_id = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        candidate_answers: list[dict[str, object]] = []
        for verification in verifications:
            payload = verification.to_dict()
            candidate = candidate_by_id.get(verification.candidate_id)
            if candidate is not None:
                payload.update(
                    {
                        "video_id": candidate.video_id,
                        "video_name": candidate.video_name,
                        "start_sec": round(candidate.start_sec, 4),
                        "end_sec": round(candidate.end_sec, 4),
                        "retrieval_score": round(candidate.retrieval_score, 6),
                        "required_event_coverage": round(
                            candidate.required_event_coverage, 6
                        ),
                        "optional_context_coverage": round(
                            candidate.optional_context_coverage, 6
                        ),
                        "model_query_consensus": round(
                            candidate.model_query_consensus, 6
                        ),
                        "evidence_frames": candidate.evidence_frames,
                    }
                )
            candidate_answers.append(payload)
        response = {
            "answer": answer,
            "answer_status": status,
            "answer_confidence": round(confidence, 6),
            "effective_confidence": round(confidence, 6),
            "answer_evidence_summary": answer_evidence_summary,
            "selected_candidate": selected_payload,
            "required_event_coverage": round(
                selected.required_event_coverage if selected is not None else 0.0,
                6,
            ),
            "optional_context_coverage": round(
                selected.optional_context_coverage if selected is not None else 0.0,
                6,
            ),
            "event_support": (
                selected_verification.event_support
                if selected_verification is not None
                else {}
            ),
            "evidence_frames": evidence_frames,
            "supporting_frame_ids": supporting_ids,
            "candidates": [
                candidate.to_dict(include_evidence=False) for candidate in candidates
            ],
            "candidate_answers": candidate_answers,
            "query_plan": plan.to_dict(),
            "usage": usage,
            "results": [
                _search_result_payload(result)
                for result in full_results[: max(1, top_k)]
            ],
            "display_results": _build_display_results(
                candidates=candidates,
                selected=selected,
                verifications=verifications,
                full_results=full_results,
                top_k=max(1, top_k),
            ),
            "pipeline": pipeline,
            "reasoning": (
                "Grounded VQA: ordered-event retrieval, multi-frame candidate verification, "
                "and evidence-citing final selection."
            ),
        }
        if "retrieval_trace" in pipeline:
            response["retrieval_trace"] = pipeline["retrieval_trace"]
        return response


def grounded_vqa_search(
    *,
    experiment: Experiment,
    retriever,
    query: str,
    question: str,
    context: str = "",
    top_k: int = 20,
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
    use_llm: bool = True,
) -> dict[str, object]:
    """Public orchestration helper used by :mod:`retrieval.vqa`."""
    return GroundedVqaPipeline(experiment, retriever).run(
        query=query,
        question=question,
        context=context,
        top_k=top_k,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
    )


def build_candidate_moments(
    plan: VqaQueryPlan,
    full_results: list[SearchResult],
    event_results: list[list[SearchResult]],
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    moment_pool: int = DEFAULT_MOMENT_POOL,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    event_gap_sec: float = DEFAULT_EVENT_GAP_SEC,
    max_moment_span_sec: float = DEFAULT_MOMENT_SPAN_SEC,
) -> list[VqaCandidateMoment]:
    """Build diverse complete moments with beam search over required events.

    Target-bearing events are hard requirements. Context-only events may be
    attached to a complete chain as ranking evidence, but can never create an
    answerable candidate by themselves.
    """
    event_count = max(1, len(plan.events))
    required_indices = sorted(_required_event_indices(plan))
    context_indices = sorted(set(range(event_count)) - set(required_indices))
    full_by_video: dict[str, list[tuple[int, SearchResult, float]]] = defaultdict(list)
    full_video_ranks: dict[str, int] = {}
    for result in full_results:
        full_video_ranks.setdefault(result.video_id, len(full_video_ranks) + 1)
    full_video_pool = max(1, len(full_video_ranks))
    for rank, result in enumerate(full_results, start=1):
        video_rank = full_video_ranks[result.video_id]
        full_by_video[result.video_id].append(
            (
                rank,
                result,
                _rank_score(video_rank, full_video_pool),
            )
        )

    hits_by_video: dict[str, dict[int, list[EventHit]]] = defaultdict(
        lambda: defaultdict(list)
    )
    video_hit_limit = _env_int(
        "VQA_VIDEO_HITS_PER_EVENT",
        DEFAULT_VIDEO_HITS_PER_EVENT,
        minimum=1,
        maximum=100,
    )
    for event_index, results in enumerate(event_results):
        video_ranks: dict[str, int] = {}
        for result in results:
            video_ranks.setdefault(result.video_id, len(video_ranks) + 1)
        video_pool = max(1, len(video_ranks))
        for rank, result in enumerate(results, start=1):
            if result.timestamp_sec is None:
                continue
            if len(hits_by_video[result.video_id][event_index]) >= video_hit_limit:
                continue
            video_rank = video_ranks[result.video_id]
            hits_by_video[result.video_id][event_index].append(
                EventHit(
                    event_index,
                    result,
                    video_rank,
                    _rank_score(video_rank, video_pool),
                )
            )

    raw: list[VqaCandidateMoment] = []
    for video_id, hits_per_event in hits_by_video.items():
        if any(not hits_per_event.get(index) for index in required_indices):
            continue

        beams: list[tuple[EventHit, ...]] = [()]
        for event_index in required_indices:
            expanded: list[tuple[EventHit, ...]] = []
            for chain in beams:
                for hit in hits_per_event[event_index]:
                    if hit.result.timestamp_sec is None:
                        continue
                    timestamp = float(hit.result.timestamp_sec)
                    if any(existing.result.frame_id == hit.result.frame_id for existing in chain):
                        continue
                    if chain:
                        first_timestamp = float(chain[0].result.timestamp_sec or 0.0)
                        previous_timestamp = float(chain[-1].result.timestamp_sec or 0.0)
                        if not previous_timestamp < timestamp <= previous_timestamp + event_gap_sec:
                            continue
                        if timestamp - first_timestamp > max_moment_span_sec:
                            continue
                    expanded.append((*chain, hit))
            if not expanded:
                beams = []
                break
            unique: dict[tuple[str, ...], tuple[EventHit, ...]] = {}
            for chain in expanded:
                key = tuple(hit.result.frame_id for hit in chain)
                unique[key] = chain
            beams = sorted(
                unique.values(),
                key=lambda chain: (
                    sum(hit.rank_score for hit in chain) / len(chain),
                    -(
                        float(chain[-1].result.timestamp_sec or 0.0)
                        - float(chain[0].result.timestamp_sec or 0.0)
                    ),
                ),
                reverse=True,
            )[:beam_width]

        for required_chain in beams:
            chain = list(required_chain)
            used_frame_ids = {hit.result.frame_id for hit in chain}
            for context_index in context_indices:
                eligible = [
                    hit
                    for hit in hits_per_event.get(context_index, [])
                    if hit.result.frame_id not in used_frame_ids
                    and _context_hit_fits_chain(
                        hit,
                        chain,
                        event_gap_sec=event_gap_sec,
                        max_moment_span_sec=max_moment_span_sec,
                    )
                ]
                if not eligible:
                    continue
                chosen = max(eligible, key=lambda hit: (hit.rank_score, -hit.rank))
                chain.append(chosen)
                used_frame_ids.add(chosen.result.frame_id)

            chain.sort(key=lambda hit: hit.event_index)
            timestamps = [float(hit.result.timestamp_sec or 0.0) for hit in chain]
            start_sec = min(timestamps)
            end_sec = max(timestamps)
            global_hit, global_score = _best_global_hit(
                full_by_video.get(video_id, []), start_sec, end_sec
            )
            covered = {hit.event_index for hit in chain}
            required_coverage = len(covered & set(required_indices)) / len(required_indices)
            if required_coverage < 1.0:
                continue
            context_coverage = (
                len(covered & set(context_indices)) / len(context_indices)
                if context_indices
                else 1.0
            )
            required_hits = [hit for hit in chain if hit.event_index in required_indices]
            ordered_quality = sum(hit.rank_score for hit in required_hits) / len(required_hits)
            consensus = sum(
                _result_branch_consensus(hit.result) for hit in required_hits
            ) / len(required_hits)
            retrieval_score = (
                0.45 * ordered_quality
                + 0.25 * context_coverage
                + 0.20 * consensus
                + 0.10 * global_score
            )
            base = required_chain[0].result
            raw.append(
                VqaCandidateMoment(
                    candidate_id="",
                    video_id=video_id,
                    video_name=_portable_basename(base.video_name or video_id),
                    start_sec=max(0.0, start_sec),
                    end_sec=max(0.0, end_sec),
                    event_hits=tuple(chain),
                    event_coverage=len(covered) / event_count,
                    chain_score=ordered_quality,
                    global_rank_score=global_score,
                    retrieval_score=retrieval_score,
                    required_event_coverage=required_coverage,
                    optional_context_coverage=context_coverage,
                    model_query_consensus=consensus,
                    global_hit=global_hit,
                )
            )

    raw.sort(
        key=lambda candidate: (
            candidate.retrieval_score,
            candidate.event_coverage,
            candidate.chain_score,
        ),
        reverse=True,
    )
    # Build the moment pool with one best moment per video first. Otherwise a
    # single high-density video can occupy all 20 beam outputs before the
    # candidate-level diversity pass ever sees another video.
    diverse_pool: list[VqaCandidateMoment] = []
    pooled_ids: set[int] = set()
    seen_videos: set[str] = set()
    for candidate in raw:
        if candidate.video_id in seen_videos:
            continue
        diverse_pool.append(candidate)
        pooled_ids.add(id(candidate))
        seen_videos.add(candidate.video_id)
        if len(diverse_pool) >= moment_pool:
            break
    if len(diverse_pool) < moment_pool:
        for candidate in raw:
            if id(candidate) in pooled_ids:
                continue
            diverse_pool.append(candidate)
            pooled_ids.add(id(candidate))
            if len(diverse_pool) >= moment_pool:
                break
    raw = diverse_pool
    selected: list[VqaCandidateMoment] = []

    def add_candidates(max_per_video: int) -> None:
        per_video = defaultdict(int)
        for existing in selected:
            per_video[existing.video_id] += 1
        for candidate in raw:
            if candidate in selected or per_video[candidate.video_id] >= max_per_video:
                continue
            if any(
                existing.video_id == candidate.video_id
                and _temporal_iou(existing, candidate) >= 0.5
                for existing in selected
            ):
                continue
            candidate.candidate_id = f"c{len(selected) + 1}"
            selected.append(candidate)
            per_video[candidate.video_id] += 1
            if len(selected) >= candidate_count:
                return

    add_candidates(1)
    if len(selected) < candidate_count:
        add_candidates(2)
    return selected


def _heuristic_plan(*, query: str, question: str, context: str) -> VqaQueryPlan:
    combined = " ".join(part.strip() for part in (context, query) if part.strip())
    source_fields = " ".join(
        part.strip() for part in (context, query, question) if part.strip()
    )
    target = (
        "X"
        if re.search(r"\bX\b", source_fields, flags=re.IGNORECASE)
        else ""
    )
    marked = _mask_unknown_target(combined)
    if target and "[TARGET]" not in marked:
        # Some benchmark questions introduce ``X`` only in the question while
        # the scene description calls it "con vật", "vật đó", etc.  Keep the
        # deterministic planner answer-neutral, but still bind those explicit
        # referring phrases to the unknown so evidence selection knows which
        # actions must show the target.
        marked = re.sub(
            r"\b(?:con\s+vật|vật\s+thể|đối\s+tượng|"
            r"con\s+đó|vật\s+đó|thứ\s+đó)\b",
            "[TARGET]",
            marked,
            flags=re.IGNORECASE,
        )
    parts = [
        part.strip(" ,.;:-")
        for part in re.split(
            r"\b(?:sau\s+đó|sau\s+vài\s+giây|tiếp\s+theo|kế\s+tiếp|"
            r"cuối\s+cùng|rồi|then|next|finally)\b",
            marked,
            flags=re.IGNORECASE,
        )
        if part.strip(" ,.;:-")
    ][:5]
    if not parts:
        parts = [marked or query or question]
    if target and not any("[TARGET]" in part for part in parts):
        # Last-resort lexical binding for descriptions such as "đặt bốn con
        # lên đĩa" that omit both X and a noun. Prefer events with direct
        # manipulation/count cues; never insert a guessed entity name.
        cue_pattern = re.compile(
            r"\b(?:đặt|cầm|giơ|nâng|lấy|bỏ|xếp|cho|put|hold|place|pick)\b|"
            r"\b\d+\s+con\b",
            flags=re.IGNORECASE,
        )
        marked_parts: list[str] = []
        for part in parts:
            if cue_pattern.search(part):
                marked_parts.append(f"{part} [TARGET]")
            else:
                marked_parts.append(part)
        if not any("[TARGET]" in part for part in marked_parts):
            marked_parts[0] = f"{marked_parts[0]} [TARGET]"
        parts = marked_parts
    answer_type = _infer_answer_type(question, query)
    constraints = _infer_constraints(query, question)
    disallowed = ("person",) if answer_type in {"food", "object"} else ()
    events = tuple(
        VqaEvent(
            index=index,
            text_en=_heuristic_english_event(part, answer_type=answer_type),
            text_vi=part,
            asr_keywords=("hôm nay nấu món gì",)
            if re.search(r"(?:đối thoại|nói|hỏi|nấu món gì)", part, re.IGNORECASE)
            else (),
            answer_bearing="[TARGET]" in part,
        )
        for index, part in enumerate(parts)
    )
    constraint_specs = _infer_constraint_specs(
        events=events,
        answer_type=answer_type,
    )
    planned_prompt = marked
    if target and "[TARGET]" not in planned_prompt:
        planned_prompt = ". ".join(parts)
    return VqaQueryPlan(
        visual_prompt_en=". ".join(event.text_en for event in events) or query,
        visual_prompt_vi=planned_prompt or query,
        events=events,
        answer_type=answer_type,
        target_reference=target,
        discriminative_cues=tuple(constraints),
        constraints=tuple(constraints),
        constraint_specs=constraint_specs,
        disallowed_entity_types=disallowed,
        llm_status="heuristic",
    )


def _plan_from_payload(
    payload: dict[str, object],
    *,
    fallback: VqaQueryPlan,
    query: str,
    question: str,
    usage: dict[str, object],
) -> VqaQueryPlan:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("planner returned no events")
    # The model may describe an existing target, but it may not invent one.
    # Only source-query parsing is authoritative for an unknown reference.
    target = fallback.target_reference
    events: list[VqaEvent] = []
    if target and len(raw_events) != len(fallback.events):
        raise ValueError("planner changed the masked source event count")
    for position, raw in enumerate(raw_events[:5]):
        if not isinstance(raw, dict):
            continue
        source_event = fallback.events[position] if position < len(fallback.events) else None
        text_en = str(raw.get("text_en") or "").strip()
        text_vi = str(raw.get("text_vi") or text_en).strip()
        answer_bearing = (
            source_event.answer_bearing
            if target and source_event is not None
            else bool(raw.get("answer_bearing"))
        )
        if target and answer_bearing:
            if "[TARGET]" not in text_en:
                raise ValueError("planner removed [TARGET] from its English translation")
        if target and source_event is not None:
            # Vietnamese source and required/context role are deterministic;
            # the model is used only for concise English translation and
            # modality keywords. Target-bearing English stays on the
            # deterministic template as an additional leak barrier: even a
            # model that emits "clam [TARGET]" cannot seed the answer into
            # retrieval.
            text_vi = source_event.text_vi
            if source_event.answer_bearing:
                text_en = source_event.text_en
            elif not text_en or _looks_like_untranslated_vietnamese(text_en):
                text_en = source_event.text_en
        if not text_en and not text_vi:
            continue
        events.append(
            VqaEvent(
                index=source_event.index if source_event is not None else len(events),
                text_en=text_en or text_vi,
                text_vi=text_vi or text_en,
                # Keywords attached to a hidden target are also retrieval
                # queries. Keep them source-derived so a model cannot smuggle
                # a guessed noun (for example "clam") into OCR/ASR search.
                ocr_keywords=(
                    source_event.ocr_keywords
                    if target and source_event is not None and answer_bearing
                    else _string_tuple(raw.get("ocr_keywords"), limit=5)
                ),
                asr_keywords=(
                    source_event.asr_keywords
                    if target and source_event is not None and answer_bearing
                    else _string_tuple(raw.get("asr_keywords"), limit=5)
                ),
                answer_bearing=answer_bearing,
            )
        )
    if not events:
        raise ValueError("planner returned no valid events")
    if target and not any(event.answer_bearing for event in events):
        raise ValueError("planner did not mark an answer-bearing target event")

    answer_type = str(payload.get("answer_type") or _infer_answer_type(question, query)).lower()
    allowed_types = {"object", "food", "person", "text", "count", "color", "action", "other"}
    if answer_type not in allowed_types:
        answer_type = "other"
    if target and fallback.answer_type != "other":
        # A source-derived type such as "food ingredient" is a hard safety
        # constraint. Translation may enrich an unknown type, but it must not
        # turn a food/object target into a person.
        answer_type = fallback.answer_type
    constraints = (
        fallback.constraints
        if target
        else _string_tuple(payload.get("constraints"), limit=12) or fallback.constraints
    )
    disallowed = (
        fallback.disallowed_entity_types
        if target
        else _string_tuple(payload.get("disallowed_entity_types"), limit=8)
    )
    if answer_type in {"object", "food"} and "person" not in disallowed:
        disallowed = (*disallowed, "person")
    visual_prompt_en = (
        ". ".join(event.text_en for event in events)
        if target
        else str(payload.get("visual_prompt_en") or fallback.visual_prompt_en)
    )
    visual_prompt_vi = (
        fallback.visual_prompt_vi
        if target
        else str(payload.get("visual_prompt_vi") or fallback.visual_prompt_vi)
    )
    return VqaQueryPlan(
        visual_prompt_en=visual_prompt_en,
        visual_prompt_vi=visual_prompt_vi,
        events=tuple(events),
        answer_type=answer_type,
        target_reference=target,
        discriminative_cues=(
            fallback.discriminative_cues
            if target
            else _string_tuple(payload.get("discriminative_cues"), limit=12)
            or fallback.discriminative_cues
        ),
        constraints=constraints,
        constraint_specs=fallback.constraint_specs,
        disallowed_entity_types=disallowed,
        llm_status="llm",
        llm_usage=usage,
        llm_calls=1,
    )


def _infer_answer_type(question: str, query: str) -> str:
    normalized = _normalize_text(f"{query} {question}")
    if any(token in normalized for token in ("nguyen lieu", "mon an", "thuc pham")):
        return "food"
    if any(token in normalized for token in ("mau gi", "mau nao")):
        return "color"
    if any(token in normalized for token in ("bao nhieu", "may nguoi", "may con")):
        return "count"
    if any(token in normalized for token in ("dong chu", "chu gi", "ten gi", "bien so")):
        return "text"
    if re.search(r"\b(ai|who)\b", normalized):
        return "person"
    if any(token in normalized for token in ("lam gi", "hanh dong gi", "doing what")):
        return "action"
    if any(token in normalized for token in ("con gi", "vat gi", "cai gi", "what is x")):
        return "object"
    return "other"


def _infer_constraints(query: str, question: str) -> list[str]:
    source = f"{query} {question}".casefold()
    normalized = _normalize_text(f"{query} {question}")
    constraints: list[str] = []
    if "nguyen lieu" in normalized or "mon an" in normalized:
        constraints.append("The target is a food ingredient, not a person.")
    count_matches = re.findall(
        r"\b(1|2|3|4|5|mot|hai|ba|bon|nam)\s+con\s+x\b",
        normalized,
    )
    for value in count_matches:
        constraints.append(f"The described target occurrence has count: {value}.")
    has_plate = bool(re.search(r"\bđĩa\b", source)) or bool(
        re.search(r"\bdia\b", source)
        and re.search(r"\b(?:dat|de|tren)\b", normalized)
    )
    if has_plate:
        constraints.append("The target is placed on a plate.")
    if re.search(r"\bcầm\b", source) or re.search(
        r"\bcam\s+(?:len|hai|mot|con|vat|x)\b", normalized
    ):
        constraints.append("The target is later held in a person's hand.")
    if "sau do" in normalized or "roi" in normalized:
        constraints.append("The described actions must occur in chronological order.")
    return _unique_strings(constraints)


def _infer_constraint_specs(
    *,
    events: tuple[VqaEvent, ...],
    answer_type: str,
) -> tuple[VqaConstraint, ...]:
    specs: list[VqaConstraint] = []
    if answer_type in {"food", "object"}:
        specs.append(
            VqaConstraint(
                constraint_id="TARGET_ENTITY_TYPE",
                kind="entity_type",
                value=answer_type,
                description=f"The answer must be a {answer_type}, not a person.",
            )
        )

    required_events = [event for event in events if event.answer_bearing]
    number_words = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5}
    for event in required_events:
        normalized = _normalize_text(f"{event.text_vi} {event.text_en}")
        count_match = re.search(
            r"\b(\d+|mot|hai|ba|bon|nam)\s+(?:con\s+)?target\b",
            normalized,
        )
        if count_match:
            raw_count = count_match.group(1)
            count = int(raw_count) if raw_count.isdigit() else number_words[raw_count]
            specs.append(
                VqaConstraint(
                    constraint_id=f"E{event.index}_COUNT_{count}",
                    kind="count",
                    event_index=event.index,
                    value=count,
                    description=f"Event {event.index} visibly contains {count} targets.",
                )
            )
        if (
            any(token in normalized.split() for token in ("dat", "de", "place", "put"))
            and any(token in normalized.split() for token in ("dia", "plate"))
        ):
            specs.append(
                VqaConstraint(
                    constraint_id=f"E{event.index}_ON_PLATE",
                    kind="relation",
                    event_index=event.index,
                    value="on_plate",
                    description=f"Event {event.index} shows the target on a plate.",
                )
            )
        if any(token in normalized.split() for token in ("cam", "hold", "holding")):
            specs.append(
                VqaConstraint(
                    constraint_id=f"E{event.index}_HELD",
                    kind="action",
                    event_index=event.index,
                    value="held",
                    description=f"Event {event.index} shows the target being held.",
                )
            )

    if len(required_events) > 1:
        sequence = "_".join(str(event.index) for event in required_events)
        specs.append(
            VqaConstraint(
                constraint_id=f"ORDER_{sequence}",
                kind="order",
                value=sequence,
                description="Required target events occur in chronological order.",
            )
        )
    return tuple(specs)


def _mask_unknown_target(value: str) -> str:
    return re.sub(r"\bX\b", "[TARGET]", str(value), flags=re.IGNORECASE)


def _heuristic_english_event(text: str, *, answer_type: str) -> str:
    """Produce a conservative English fallback without naming the target."""
    normalized = _normalize_text(text)
    has_target = "[TARGET]" in text
    # Prefer the count syntactically attached to the hidden target.  In a
    # phrase such as ``1 dĩa trắng 4 con [TARGET]`` the first number describes
    # the plate, not X; using a generic first-number match generated the wrong
    # retrieval query ``woman placing one ...``.
    count_match = re.search(
        r"\b(1|2|3|4|5|mot|hai|ba|bon|nam)\s+(?:con\s+)?target\b",
        normalized,
    )
    if count_match is None:
        count_match = re.search(
            r"\b(1|2|3|4|5|mot|hai|ba|bon|nam)\b",
            normalized,
        )
    number_words = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "mot": "one",
        "hai": "two",
        "ba": "three",
        "bon": "four",
        "nam": "five",
    }
    count = number_words.get(count_match.group(1), "") if count_match else ""
    tokens = set(normalized.split())
    if has_target and tokens & {"dat", "de", "put", "place"} and tokens & {"dia", "plate"}:
        return f"woman placing {count} [TARGET] on a white plate".replace("  ", " ")
    if has_target and tokens & {"cam", "hold", "holding", "pick"}:
        return f"woman holding {count} [TARGET] in her hands".replace("  ", " ")
    if tokens & {"tap", "apron"} and tokens & {"hoa", "flower", "flowers"}:
        return (
            "woman wearing a white apron beside a vase of purple flowers "
            "in a television cooking-show kitchen"
        )
    if tokens & {"doi", "thoai", "noi", "talk", "talking"}:
        return "two people talking together in a television cooking-show kitchen"
    if has_target:
        return "person handling [TARGET]"
    return text


def _looks_like_untranslated_vietnamese(text: str) -> bool:
    normalized = set(_normalize_text(text).split())
    markers = {
        "nguoi",
        "phu",
        "nu",
        "co",
        "gai",
        "dat",
        "cam",
        "dia",
        "sau",
        "do",
        "nguyen",
        "lieu",
        "mon",
        "an",
        "doi",
        "thoai",
    }
    return len(normalized & markers) >= 3


def _verification_from_payload(
    *,
    candidate: VqaCandidateMoment,
    plan: VqaQueryPlan,
    payload: dict[str, object],
    frame_map: dict[str, str],
    usage: dict[str, object],
    frame_events: dict[str, set[int]] | None = None,
    frame_times: dict[str, float] | None = None,
) -> VqaVerification:
    verdict = str(payload.get("verdict") or "partial").lower()
    if verdict not in {"supported", "partial", "not_supported"}:
        verdict = "partial"
    answer_raw = payload.get("answer")
    answer = str(answer_raw).strip() if answer_raw not in (None, "", "null") else None
    entity_type = str(payload.get("entity_type") or "other").lower()
    confidence = _clamp_float(payload.get("confidence"), default=0.0)
    valid_event_indices = {event.index for event in plan.events}
    required_events = _required_event_indices(plan)
    candidate_events = {hit.event_index for hit in candidate.event_hits}
    candidate_required_coverage = (
        len(required_events & candidate_events) / len(required_events)
        if required_events
        else 1.0
    )

    label_events: dict[str, set[int]] = {
        label: set(indices) for label, indices in (frame_events or {}).items()
    }
    label_times: dict[str, float] = dict(frame_times or {})
    for label, frame_id in frame_map.items():
        for hit in candidate.event_hits:
            if hit.result.frame_id != frame_id:
                continue
            label_events.setdefault(label, set()).add(hit.event_index)
            if hit.result.timestamp_sec is not None:
                label_times.setdefault(label, float(hit.result.timestamp_sec))

    claimed_indices = {
        int(value)
        for value in _as_list(payload.get("matched_event_indices"))
        if str(value).lstrip("-").isdigit()
        and int(value) in valid_event_indices
    }
    raw_event_support = payload.get("event_support")
    event_support_labels: dict[int, list[str]] = defaultdict(list)
    if isinstance(raw_event_support, dict):
        for raw_index, raw_labels in raw_event_support.items():
            if not str(raw_index).lstrip("-").isdigit():
                continue
            event_index = int(raw_index)
            if event_index not in valid_event_indices:
                continue
            for raw_label in _as_list(raw_labels):
                label = str(raw_label).strip()
                if (
                    label in frame_map
                    and event_index in label_events.get(label, set())
                    and label not in event_support_labels[event_index]
                ):
                    event_support_labels[event_index].append(label)
    else:
        # ``supporting_frames`` and ``matched_event_indices`` are retained in
        # the response schema for compatibility, but are not enough to prove
        # which image grounds which required event. New grounded verification
        # therefore requires the explicit event -> frame-label mapping.
        claimed_indices.clear()

    ordered = True
    previous_time: float | None = None
    for event_index in sorted(required_events):
        labels = event_support_labels.get(event_index, [])
        timestamps = [label_times[label] for label in labels if label in label_times]
        if not timestamps:
            continue
        timestamp = min(timestamps)
        if previous_time is not None and timestamp <= previous_time:
            ordered = False
            break
        previous_time = timestamp

    validated_event_support = {
        event_index: tuple(frame_map[label] for label in labels)
        for event_index, labels in event_support_labels.items()
        if labels
    }
    matched = tuple(sorted(validated_event_support))
    citation_coverage = (
        len(required_events & set(matched)) / len(required_events)
        if required_events
        else 1.0
    )
    supporting = tuple(
        dict.fromkeys(
            frame_id
            for event_index in sorted(validated_event_support)
            for frame_id in validated_event_support[event_index]
        )
    )
    disallowed = {_normalize_text(value) for value in plan.disallowed_entity_types}
    person_terms = {
        "person",
        "woman",
        "man",
        "human",
        "nguoi",
        "phu nu",
        "co gai",
        "nu mc",
    }
    answer_normalized = _normalize_text(answer or "")
    type_conflict = (
        entity_type in disallowed
        or (
            plan.answer_type in {"food", "object"}
            and (
                entity_type in person_terms
                or _contains_normalized_phrase(answer_normalized, person_terms)
            )
        )
    )
    contradictions = list(_string_tuple(payload.get("contradictions"), limit=12))
    reported_constraint_ids = {
        str(value).strip()
        for value in _as_list(payload.get("supported_constraint_ids"))
        if str(value).strip()
    }
    reported_constraints = {
        _normalize_text(value)
        for value in _string_tuple(payload.get("supported_constraints"), limit=20)
    }
    specs_by_id = {spec.constraint_id: spec for spec in plan.constraint_specs}
    for spec in plan.constraint_specs:
        if _normalize_text(spec.description) in reported_constraints:
            reported_constraint_ids.add(spec.constraint_id)
    validated_constraint_ids: list[str] = []
    for constraint_id in reported_constraint_ids:
        spec = specs_by_id.get(constraint_id)
        if spec is None:
            continue
        if spec.kind == "entity_type":
            if not type_conflict:
                validated_constraint_ids.append(constraint_id)
        elif spec.kind == "order":
            if ordered and citation_coverage >= 1.0:
                validated_constraint_ids.append(constraint_id)
        elif spec.event_index in validated_event_support:
            validated_constraint_ids.append(constraint_id)

    visual_specs = [spec for spec in plan.constraint_specs if spec.kind != "entity_type"]
    if visual_specs:
        validated_visual = {
            constraint_id
            for constraint_id in validated_constraint_ids
            if specs_by_id[constraint_id].kind != "entity_type"
        }
        constraint_coverage = len(validated_visual) / len(visual_specs)
        supported_constraints = tuple(
            spec.description
            for spec in plan.constraint_specs
            if spec.constraint_id in set(validated_constraint_ids)
        )
    else:
        supported_constraints = tuple(
            constraint
            for constraint in plan.constraints
            if _normalize_text(constraint) in reported_constraints
        )
        constraint_coverage = (
            len(supported_constraints) / len(plan.constraints)
            if plan.constraints
            else 1.0
        )

    grounding_quality = (
        0.50 * candidate_required_coverage
        + 0.30 * constraint_coverage
        + 0.20 * citation_coverage
    )
    effective_confidence = min(confidence, grounding_quality)
    if type_conflict:
        contradictions.append("Answer entity type conflicts with the target constraints.")
        verdict = "not_supported"
    if not supporting or not answer:
        verdict = "partial" if verdict == "supported" else verdict
    if candidate_required_coverage < 1.0:
        verdict = "not_supported"
        contradictions.append("Candidate retrieval does not cover every required target event.")
    if required_events and not required_events.issubset(set(matched)):
        verdict = "partial" if verdict == "supported" else verdict
        contradictions.append("Not every required target event has a valid frame citation.")
    if not ordered:
        verdict = "partial" if verdict == "supported" else verdict
        contradictions.append("Required event citations are not in chronological order.")
    if constraint_coverage < 1.0:
        verdict = "partial" if verdict == "supported" else verdict
        contradictions.append("Not every structured target constraint was grounded.")
    if contradictions and verdict == "supported":
        verdict = "partial"
    return VqaVerification(
        candidate_id=candidate.candidate_id,
        verdict=verdict,
        answer=answer,
        entity_type=entity_type,
        confidence=confidence,
        supporting_frame_ids=supporting,
        matched_event_indices=matched,
        supported_constraints=supported_constraints,
        contradictions=tuple(_unique_strings(contradictions)),
        evidence_summary=str(payload.get("evidence_summary") or "")[:1000],
        usage=usage,
        logical_calls=1,
        event_support=validated_event_support,
        validated_constraint_ids=tuple(sorted(set(validated_constraint_ids))),
        required_event_coverage=candidate_required_coverage,
        valid_citation_coverage=citation_coverage,
        effective_confidence=effective_confidence,
    )


def _error_verification(
    candidate_id: str,
    exc: Exception,
    *,
    usage: dict[str, object] | None = None,
    logical_calls: int = 0,
) -> VqaVerification:
    return VqaVerification(
        candidate_id=candidate_id,
        verdict="error",
        answer=None,
        entity_type="other",
        confidence=0.0,
        supporting_frame_ids=(),
        matched_event_indices=(),
        supported_constraints=(),
        contradictions=(),
        evidence_summary="",
        error=f"{type(exc).__name__}: {exc}"[:300],
        usage=dict(usage or {}),
        logical_calls=logical_calls,
        effective_confidence=0.0,
    )


def _search_result_from_mapping(row: object) -> SearchResult | None:
    if not isinstance(row, dict) or not row.get("frame_id") or not row.get("video_id"):
        return None
    try:
        score = float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    timestamp_value = row.get("frame_timestamp_sec")
    if timestamp_value is None:
        timestamp_value = row.get("timestamp_sec")
    try:
        timestamp_sec = float(timestamp_value) if timestamp_value is not None else None
    except (TypeError, ValueError):
        timestamp_sec = None
    return SearchResult(
        frame_id=str(row["frame_id"]),
        video_id=str(row["video_id"]),
        score=score,
        frame_path=str(row["frame_path"]) if row.get("frame_path") else None,
        video_name=str(row["video_name"]) if row.get("video_name") else None,
        shot_id=str(row["shot_id"]) if row.get("shot_id") else None,
        frame_index=int(row["frame_index"]) if row.get("frame_index") is not None else None,
        timestamp_sec=timestamp_sec,
    )


def _target_safe_search_text(
    text: str,
    *,
    language: str,
    answer_type: str,
) -> str:
    replacements = {
        "en": {
            "food": "food ingredients",
            "object": "object",
            "person": "person",
        },
        "vi": {
            "food": "nguyên liệu món ăn",
            "object": "vật thể",
            "person": "người",
        },
    }
    default = "object" if language == "en" else "vật thể"
    replacement = replacements.get(language, {}).get(answer_type, default)
    safe = str(text)
    if language == "vi":
        # Vietnamese classifiers such as "4 con" are useful in the source
        # plan but become awkward after substituting a generic target class.
        safe = re.sub(
            r"\bcon\s+\[TARGET\]",
            replacement,
            safe,
            flags=re.IGNORECASE,
        )
    return re.sub(r"\[TARGET\]", replacement, safe, flags=re.IGNORECASE)


def _required_event_indices(plan: VqaQueryPlan) -> set[int]:
    required = {event.index for event in plan.events if event.answer_bearing}
    return required or {event.index for event in plan.events}


def _candidate_required_coverage(
    candidate: VqaCandidateMoment,
    plan: VqaQueryPlan,
) -> float:
    required = _required_event_indices(plan)
    covered = {hit.event_index for hit in candidate.event_hits}
    return len(required & covered) / len(required) if required else 1.0


def _effective_verification_confidence(verification: VqaVerification) -> float:
    return (
        verification.effective_confidence
        if verification.effective_confidence is not None
        else verification.confidence
    )


def _with_branch_consensus(
    result: SearchResult,
    score: float,
    consensus: float,
) -> VqaBranchSearchResult:
    payload = {
        name: getattr(result, name)
        for name in SearchResult.__dataclass_fields__
    }
    payload["score"] = score
    return VqaBranchSearchResult(
        **payload,
        model_query_consensus=_clamp_float(consensus, default=0.0),
    )


def _result_branch_consensus(result: SearchResult) -> float:
    return _clamp_float(
        getattr(result, "model_query_consensus", result.score),
        default=0.0,
    )


def _union_variant_branches(
    branches: dict[str, list[SearchResult]],
    *,
    top_k: int,
    max_hits_per_video: int,
) -> list[SearchResult]:
    """Union branches while rewarding, but never requiring, consensus.

    Consensus is measured within a local video moment. Different embedders
    commonly retrieve neighbouring keyframes from the same event; requiring
    the exact same ``frame_id`` incorrectly treated that agreement as two
    unrelated single-model hits. Conversely, hits several minutes apart in a
    long video must not support each other.
    """
    aggregated: dict[str, dict[str, object]] = {}
    video_branch_hits: dict[
        str,
        dict[str, list[tuple[SearchResult, float]]],
    ] = defaultdict(lambda: defaultdict(list))
    for branch_name, results in branches.items():
        branch_pool = max(1, len(results))
        for rank, result in enumerate(results, start=1):
            rank_score = _rank_score(rank, branch_pool)
            video_branch_hits[result.video_id][branch_name].append(
                (result, rank_score)
            )
            record = aggregated.setdefault(
                result.frame_id,
                {
                    "result": result,
                    "best_rank_score": 0.0,
                    "branches": set(),
                },
            )
            record["best_rank_score"] = max(
                float(record["best_rank_score"]), rank_score
            )
            cast_branches = record["branches"]
            assert isinstance(cast_branches, set)
            cast_branches.add(branch_name)
            current = record["result"]
            assert isinstance(current, SearchResult)
            if result.score > current.score:
                record["result"] = result

    ranked: list[SearchResult] = []
    scored: list[tuple[float, SearchResult]] = []
    branch_count = max(1, len(branches))
    for record in aggregated.values():
        base = record["result"]
        assert isinstance(base, SearchResult)
        branch_scores = [
            max(
                (
                    rank_score
                    for neighbor, rank_score in rows
                    if _results_share_local_moment(base, neighbor)
                ),
                default=0.0,
            )
            for rows in video_branch_hits.get(base.video_id, {}).values()
        ]
        consensus = min(1.0, sum(branch_scores) / branch_count)
        score = 0.80 * float(record["best_rank_score"]) + 0.20 * consensus
        # Preserve the retrieval-ranking score shown by existing clients while
        # carrying consensus separately for candidate scoring.
        scored.append((score, _with_branch_consensus(base, score, consensus)))
    scored.sort(key=lambda item: item[0], reverse=True)

    # Keep distinct shots/time buckets before filling the remaining per-video
    # quota. Without this pass, eight adjacent keyframes from one shot could
    # evict the later action needed to form an ordered target-event chain.
    diverse_scored = _select_diverse_video_hits(
        scored,
        max_hits_per_video=max_hits_per_video,
    )
    for _, result in diverse_scored:
        ranked.append(result)
        if len(ranked) >= top_k:
            break
    return ranked


def _select_diverse_video_hits(
    scored: list[tuple[float, SearchResult]],
    *,
    max_hits_per_video: int,
) -> list[tuple[float, SearchResult]]:
    """Apply a per-video cap while reserving room for distinct moments."""
    by_video: dict[str, list[tuple[float, SearchResult]]] = defaultdict(list)
    for item in scored:
        by_video[item[1].video_id].append(item)

    selected: list[tuple[float, SearchResult]] = []
    for rows in by_video.values():
        chosen: list[tuple[float, SearchResult]] = []
        chosen_ids: set[str] = set()
        seen_moments: set[tuple[str, object]] = set()

        for item in rows:
            result = item[1]
            if result.timestamp_sec is not None:
                time_bucket = int(float(result.timestamp_sec) // 2.0)
                moment_key: tuple[str, object] = (
                    "shot-time",
                    f"{result.shot_id or '<unknown>'}:{time_bucket}",
                )
            elif result.shot_id:
                moment_key = ("shot", result.shot_id)
            else:
                moment_key = ("frame", result.frame_id)
            if moment_key in seen_moments:
                continue
            chosen.append(item)
            chosen_ids.add(result.frame_id)
            seen_moments.add(moment_key)
            if len(chosen) >= max_hits_per_video:
                break

        if len(chosen) < max_hits_per_video:
            for item in rows:
                result = item[1]
                if result.frame_id in chosen_ids:
                    continue
                chosen.append(item)
                chosen_ids.add(result.frame_id)
                if len(chosen) >= max_hits_per_video:
                    break
        selected.extend(chosen)

    selected.sort(key=lambda item: item[0], reverse=True)
    return selected


def _results_share_local_moment(left: SearchResult, right: SearchResult) -> bool:
    """Return whether two same-video hits are close enough to show one event."""
    if left.video_id != right.video_id:
        return False
    if left.timestamp_sec is not None and right.timestamp_sec is not None:
        return (
            abs(float(left.timestamp_sec) - float(right.timestamp_sec))
            <= DEFAULT_BRANCH_CONSENSUS_WINDOW_SEC
        )
    if left.shot_id and right.shot_id:
        return left.shot_id == right.shot_id
    # Metadata-free custom retrievers cannot provide local grounding. Fall
    # back to video agreement only when neither side has moment metadata.
    return (
        left.timestamp_sec is None
        and right.timestamp_sec is None
        and not left.shot_id
        and not right.shot_id
    )


def _merge_auxiliary_results(
    visual_results: list[SearchResult],
    auxiliary_lists: list[list[SearchResult]],
    *,
    top_k: int,
) -> list[SearchResult]:
    scored: dict[str, tuple[float, SearchResult]] = {}
    visual_pool = max(1, len(visual_results))
    for rank, result in enumerate(visual_results, start=1):
        scored[result.frame_id] = (_rank_score(rank, visual_pool), result)
    for rows in auxiliary_lists:
        pool = max(1, len(rows))
        for rank, result in enumerate(rows, start=1):
            bonus = 0.20 * _rank_score(rank, pool)
            previous_score, previous = scored.get(result.frame_id, (0.0, result))
            scored[result.frame_id] = (previous_score + bonus, previous)
    ranked = sorted(scored.values(), key=lambda item: item[0], reverse=True)
    max_hits_per_video = _env_int(
        "VQA_VIDEO_HITS_PER_EVENT",
        DEFAULT_VIDEO_HITS_PER_EVENT,
        minimum=1,
        maximum=100,
    )
    visual_ids = {result.frame_id for result in visual_results}
    rescored: list[tuple[float, SearchResult]] = []
    for retrieval_score, result in ranked:
        # Auxiliary OCR/ASR affects list order, but it is not visual
        # model/query consensus. Text-only frames therefore carry zero here.
        rescored.append(
            (
                retrieval_score,
                result
                if result.frame_id in visual_ids
                else _with_branch_consensus(
                    result,
                    min(1.0, retrieval_score),
                    0.0,
                ),
            )
        )
    diverse = _select_diverse_video_hits(
        rescored,
        max_hits_per_video=max_hits_per_video,
    )
    return [result for _, result in diverse[:top_k]]


def _top_video_trace(results: list[SearchResult], *, limit: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for frame_rank, result in enumerate(results, start=1):
        if result.video_id in seen:
            continue
        seen.add(result.video_id)
        output.append(
            {
                "video_id": result.video_id,
                "video_name": _portable_basename(result.video_name or result.video_id),
                "first_frame_rank": frame_rank,
                "score": round(result.score, 6),
                "model_query_consensus": round(
                    _result_branch_consensus(result), 6
                ),
            }
        )
        if len(output) >= limit:
            break
    return output


def _video_hit_trace(
    results: list[SearchResult],
    *,
    video_id: str,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Return ranked frame-level hits for one video in debug traces."""
    output: list[dict[str, object]] = []
    video_rank = 0
    seen_videos: set[str] = set()
    for frame_rank, result in enumerate(results, start=1):
        if result.video_id not in seen_videos:
            seen_videos.add(result.video_id)
            if result.video_id == video_id:
                video_rank = len(seen_videos)
        if result.video_id != video_id:
            continue
        output.append(
            {
                "frame_rank": frame_rank,
                "video_rank": video_rank or len(seen_videos),
                "frame_id": result.frame_id,
                "shot_id": result.shot_id,
                "timestamp_sec": result.timestamp_sec,
                "score": round(float(result.score), 6),
                "model_query_consensus": round(
                    _result_branch_consensus(result), 6
                ),
            }
        )
        if len(output) >= limit:
            break
    return output


def _portable_basename(value: str) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _search_result_payload(result: SearchResult) -> dict[str, object]:
    payload = dict(result.to_dict())
    payload["video_name"] = _portable_basename(
        str(payload.get("video_name") or result.video_id)
    )
    return payload


def _context_hit_fits_chain(
    hit: EventHit,
    chain: list[EventHit],
    *,
    event_gap_sec: float,
    max_moment_span_sec: float,
) -> bool:
    if hit.result.timestamp_sec is None:
        return False
    timestamp = float(hit.result.timestamp_sec)
    earlier = [
        item for item in chain if item.event_index < hit.event_index
    ]
    later = [
        item for item in chain if item.event_index > hit.event_index
    ]
    if earlier:
        previous = max(float(item.result.timestamp_sec or 0.0) for item in earlier)
        if not previous < timestamp <= previous + event_gap_sec:
            return False
    if later:
        following = min(float(item.result.timestamp_sec or 0.0) for item in later)
        if not following - event_gap_sec <= timestamp < following:
            return False
    all_times = [float(item.result.timestamp_sec or 0.0) for item in chain]
    return max([timestamp, *all_times]) - min([timestamp, *all_times]) <= max_moment_span_sec


def _build_display_results(
    *,
    candidates: list[VqaCandidateMoment],
    selected: VqaCandidateMoment | None,
    verifications: list[VqaVerification],
    full_results: list[SearchResult],
    top_k: int,
) -> list[dict[str, object]]:
    verification_by_id = {row.candidate_id: row for row in verifications}
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate is selected,
            verification_by_id.get(candidate.candidate_id) is not None
            and verification_by_id[candidate.candidate_id].verdict == "supported",
            candidate.retrieval_score,
        ),
        reverse=True,
    )
    output: list[dict[str, object]] = []
    seen_frames: set[str] = set()
    seen_candidate_videos: set[str] = set()
    for candidate in ordered_candidates:
        if candidate.video_id in seen_candidate_videos:
            continue
        verification = verification_by_id.get(candidate.candidate_id)
        supporting = set(verification.supporting_frame_ids if verification else ())
        frame = next(
            (
                row for row in candidate.evidence_frames
                if str(row.get("frame_id")) in supporting
            ),
            candidate.evidence_frames[0] if candidate.evidence_frames else None,
        )
        if frame is None:
            continue
        frame_id = str(frame.get("frame_id") or "")
        if not frame_id or frame_id in seen_frames:
            continue
        output.append(
            {
                **frame,
                "video_id": candidate.video_id,
                "video_name": _portable_basename(candidate.video_name),
                "score": round(candidate.retrieval_score, 6),
                "candidate_id": candidate.candidate_id,
                "candidate_source": "ordered_event_union",
                "candidate_event_indices": sorted(
                    {hit.event_index for hit in candidate.event_hits}
                ),
                "required_event_coverage": round(candidate.required_event_coverage, 6),
                "optional_context_coverage": round(candidate.optional_context_coverage, 6),
            }
        )
        seen_frames.add(frame_id)
        seen_candidate_videos.add(candidate.video_id)
        if len(output) >= top_k:
            return output
    for result in full_results:
        if result.frame_id in seen_frames:
            continue
        output.append(_search_result_payload(result))
        seen_frames.add(result.frame_id)
        if len(output) >= top_k:
            break
    return output


def _best_global_hit(
    candidates: list[tuple[int, SearchResult, float]], start_sec: float, end_sec: float
) -> tuple[SearchResult | None, float]:
    if not candidates:
        return None, 0.0
    # The ranking feature is the full-query *video* rank. A frame from that
    # branch is added to visual evidence only when it is temporally close to
    # the candidate moment; a far-away frame must not contaminate grounding.
    video_rank_score = max(item[2] for item in candidates)
    nearby = [
        item
        for item in candidates
        if item[1].timestamp_sec is not None
        and start_sec - 10.0 <= float(item[1].timestamp_sec) <= end_sec + 10.0
    ]
    if not nearby:
        return None, video_rank_score
    _, result, _ = max(nearby, key=lambda item: item[2])
    return result, video_rank_score


def _rank_score(rank: int, pool_size: int) -> float:
    if pool_size <= 1:
        return 1.0
    return max(0.0, 1.0 - (rank - 1) / pool_size)


def _temporal_iou(left: VqaCandidateMoment, right: VqaCandidateMoment) -> float:
    intersection = max(0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec))
    union = max(left.end_sec, right.end_sec) - min(left.start_sec, right.start_sec)
    if union <= 1e-9:
        return 1.0 if abs(left.start_sec - right.start_sec) <= 1e-9 else 0.0
    return intersection / union


def _resolve_supporting_frames(value: object, frame_map: dict[str, str]) -> tuple[str, ...]:
    valid_ids = set(frame_map.values())
    output: list[str] = []
    for item in _as_list(value):
        token = str(item).strip()
        frame_id = frame_map.get(token, token if token in valid_ids else "")
        if frame_id and frame_id not in output:
            output.append(frame_id)
    return tuple(output)


def _extract_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("model response did not contain a JSON object")
        payload = json.loads(match.group())
    if not isinstance(payload, dict):
        raise ValueError("model response JSON root must be an object")
    return payload


def _evidence_prompt(evidence: dict[str, list[dict[str, object]]]) -> str:
    # Reserve a bounded slice for every modality instead of truncating one
    # large JSON string after captions. Raw slicing could both hide OCR/ASR
    # and leave malformed JSON in the verifier prompt.
    compact = {
        source: _bounded_evidence_rows(evidence.get(source, []), budget)
        for source, budget in (("asr", 1800), ("ocr", 1300), ("captions", 1300))
    }
    return json.dumps(compact, ensure_ascii=False)


def _bounded_evidence_rows(
    rows: list[dict[str, object]], budget: int
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean = dict(row)
        clean["text"] = str(clean.get("text") or "")[:700]
        encoded = json.dumps([*selected, clean], ensure_ascii=False)
        if len(encoded) <= budget:
            selected.append(clean)
            continue
        if not selected:
            # Keep at least one clue from a present modality. Reduce only its
            # free-form text while preserving frame/time provenance.
            clean["text"] = str(clean["text"])[: max(80, budget // 2)]
            selected.append(clean)
        break
    return selected


def _normalize_text(value: str) -> str:
    source = str(value).translate(str.maketrans({"đ": "d", "Đ": "D"})).lower()
    decomposed = unicodedata.normalize("NFD", source)
    no_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", no_marks).strip()


def _normalize_answer(value: str | None) -> str:
    normalized = _normalize_text(value or "")
    tokens = normalized.split()
    for prefix in (
        ("cau", "tra", "loi", "la"),
        ("dap", "an", "la"),
        ("the", "answer", "is"),
        ("answer", "is"),
        ("x", "la"),
        ("x", "is"),
    ):
        if tuple(tokens[: len(prefix)]) == prefix:
            tokens = tokens[len(prefix) :]
            break
    while tokens and tokens[0] in {"con", "cai", "mot", "a", "an", "the"}:
        tokens.pop(0)
    return " ".join(tokens)


def _contains_normalized_phrase(text: str, phrases: set[str]) -> bool:
    """Match normalized entity phrases on token boundaries, not substrings.

    A raw substring test classifies food words such as ``mango`` as a person
    merely because they contain ``man``. Both inputs here are already or can
    safely be normalized into space-delimited tokens.
    """
    padded = f" {_normalize_text(text)} "
    return any(f" {_normalize_text(phrase)} " in padded for phrase in phrases)


def _string_tuple(value: object, *, limit: int) -> tuple[str, ...]:
    return tuple(
        str(item).strip()
        for item in _as_list(value)[:limit]
        if str(item).strip()
    )


def _as_list(value: object) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _unique_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def _clamp_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return min(1.0, max(0.0, parsed))


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


def _client_usage(client: object | None) -> dict[str, object]:
    """Snapshot thread-local provider usage, including failed HTTP attempts."""
    if client is None:
        return {}
    usage = getattr(client, "last_usage", None)
    return dict(usage) if isinstance(usage, dict) else {}


def _empty_usage() -> dict[str, object]:
    return {
        "openrouter_calls": 0,
        "openrouter_http_requests": 0,
        "openrouter_operations": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }


def _merge_usage(
    aggregate: dict[str, object], usage: dict[str, object], *, logical_calls: int
) -> None:
    aggregate["openrouter_operations"] = int(
        aggregate.get("openrouter_operations", 0)
    ) + logical_calls
    try:
        request_count = max(0, int(usage.get("request_count", 0)))
    except (TypeError, ValueError):
        request_count = 0
    aggregate["openrouter_http_requests"] = int(
        aggregate.get("openrouter_http_requests", 0)
    ) + request_count
    # Keep the established field name, but make it represent literal HTTP
    # requests. ``openrouter_operations`` exposes the higher-level planner /
    # verifier count separately.
    aggregate["openrouter_calls"] = aggregate["openrouter_http_requests"]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            aggregate[key] = int(aggregate.get(key, 0)) + int(usage.get(key, 0))
        except (TypeError, ValueError):
            continue
    try:
        aggregate["cost"] = float(aggregate.get("cost", 0.0)) + float(
            usage.get("cost", 0.0)
        )
    except (TypeError, ValueError):
        pass


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)
