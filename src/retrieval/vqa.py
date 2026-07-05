"""Pipeline orchestrator — kết nối CLIP search + Temporal Search + Agent.

4 pipeline theo tài liệu training:
  - Textual KIS: Query → Search → Temporal Search → Reranking → Submission
  - Video KIS:   (chưa implement)
  - VQA:         Query → Search → Temporal Search → Reranking → Shot Validation → Agent → Answer
  - TRAKE:       Query → Search → Temporal Search → Reranking → N events → Submission
"""

from __future__ import annotations

import logging

import numpy as np

from config.settings import Experiment
from modules.embedding import build_embedder
from modules.reranker.base import Reranker, build_reranker
from retrieval import build_retriever
from retrieval.temporal_search import (
    ShotValidator,
    find_segments,
    gather_frame_s,
    load_temporal_data,
)
from agent import create_agent

LOGGER = logging.getLogger(__name__)


def _run_temporal_pipeline(
    experiment: Experiment,
    query: str,
    top_k: int = 20,
    reranker: Reranker | None = None,
    reranker_top_k: int = 10,
) -> dict:
    """Run CLIP search, optional reranking, and temporal expansion.

    Shared by VQA and TRAKE pipelines.

    Args:
        experiment:     Active experiment.
        query:          Text retrieval query.
        top_k:          Candidates retrieved from Qdrant (first-stage pool).
        reranker:       Optional cross-encoder reranker. ``None`` skips reranking.
        reranker_top_k: Candidates kept after reranking (second-stage output).
    """
    pipeline_stages = {}

    # Stage 1: fast bi-encoder retrieval via SigLIP + Qdrant.
    retriever = build_retriever(experiment)
    clip_results = retriever.search(query=query, top_k=top_k)
    pipeline_stages["clip_search"] = {
        "top_k": top_k,
        "results_count": len(clip_results),
    }

    # Stage 2 (optional): cross-encoder reranking over the first-stage pool.
    if reranker is not None and clip_results:
        LOGGER.info("Reranker: scoring %d candidates...", len(clip_results))
        clip_results = reranker.rerank(query=query, results=clip_results)
        clip_results = clip_results[:reranker_top_k]
        pipeline_stages["rerank"] = {
            "model": getattr(reranker, "model_name", "unknown"),
            "reranker_top_k": reranker_top_k,
            "results_after_rerank": len(clip_results),
        }
        LOGGER.info("Reranker: kept %d results after reranking.", len(clip_results))

    if not clip_results:
        return {
            "pipeline": pipeline_stages,
            "clip_results": [],
            "segments": [],
            "frame_embeddings": None,
            "frame_records": [],
            "query_embedding": None,
        }

    # Load temporal data
    try:
        frame_embeddings, frame_records = load_temporal_data(experiment.run_dir)
    except FileNotFoundError as exc:
        pipeline_stages["error"] = str(exc)
        return {
            "pipeline": pipeline_stages,
            "clip_results": clip_results,
            "segments": [],
            "frame_embeddings": None,
            "frame_records": [],
            "query_embedding": None,
        }

    # Map CLIP results to positions in sorted embeddings
    hit_positions = set()
    for r in clip_results:
        for i, rec in enumerate(frame_records):
            if rec.get("frame_id") == r.frame_id:
                hit_positions.add(i)
                break

    if not hit_positions:
        pipeline_stages["temporal_search"] = {"error": "No matching positions found"}
        return {
            "pipeline": pipeline_stages,
            "clip_results": clip_results,
            "segments": [],
            "frame_embeddings": frame_embeddings,
            "frame_records": frame_records,
            "query_embedding": None,
        }

    # Temporal search từ mỗi vị trí
    sorted_hits = sorted(hit_positions)
    segments = find_segments(
        start_indices=list(hit_positions),
        frame_embeddings=frame_embeddings,
        frame_records=frame_records,
        tolerance_threshold=3,
        min_gap=2,
    )
    pipeline_stages["temporal_search"] = {
        "positions_checked": len(sorted_hits),
        "segments_found": len(segments),
    }

    # Query embedding cho shot validation
    embedder = build_embedder(
        model_name=experiment.config.embedding_model,
        device=experiment.config.device,
    )
    query_embedding = embedder.embed_text(query)

    # Gather shot cho mỗi segment
    shots = []
    for seg in segments:
        shot = gather_frame_s(seg, frame_records)
        if shot and shot.frame_count > 0:
            # Gán video_name từ clip result đầu tiên
            if clip_results:
                shot.video_name = clip_results[0].video_name or ""
            shots.append((shot, seg))

    pipeline_stages["gather_shot"] = {
        "shots_count": len(shots),
    }

    return {
        "pipeline": pipeline_stages,
        "clip_results": clip_results,
        "segments": segments,
        "shots": shots,
        "frame_embeddings": frame_embeddings,
        "frame_records": frame_records,
        "query_embedding": np.asarray(query_embedding),
    }


def vqa_search(
    experiment: Experiment,
    query: str,
    question: str,
    context: str = "",
    top_k: int = 20,
    reranker=None,
    reranker_top_k: int = 10,
    vqa_backend: str = "gemini",
) -> dict:
    """VQA pipeline: CLIP search → (rerank) → temporal → shot validation → agent answer.

    Args:
        experiment:     Active experiment.
        query:          Visual search query (describes the scene to locate).
        question:       VQA question forwarded to the agent (e.g. ``"What is the plate number?"``).
        context:        Optional scene context prepended to the retrieval query.
        top_k:          First-stage candidate pool size (Qdrant).
        reranker:       Optional cross-encoder reranker; ``None`` skips reranking.
        reranker_top_k: Candidates retained after reranking.

    Returns:
        Dict with keys ``answer``, ``results``, ``pipeline``.
    """
    retrieval_text = f"{context} {query}".strip()
    data = _run_temporal_pipeline(experiment, retrieval_text, top_k, reranker=reranker, reranker_top_k=reranker_top_k)
    pipeline_stages = data.get("pipeline", {})
    clip_results = data.get("clip_results", [])
    shots = data.get("shots", [])
    frame_embeddings = data.get("frame_embeddings")
    frame_records = data.get("frame_records", [])
    query_embedding = data.get("query_embedding")

    if not clip_results:
        return {
            "answer": "No relevant frames found.",
            "results": [],
            "pipeline": pipeline_stages,
            "reasoning": "",
        }

    shots = data.get("shots", [])
    if not shots:
        # Fallback: dùng clip result đầu tiên
        return {
            "answer": "Could not find a valid shot segment.",
            "results": [r.to_dict() for r in clip_results[:5]],
            "pipeline": pipeline_stages,
            "reasoning": "Temporal search found no segments.",
        }

    # Shot validation: chọn shot tốt nhất (based on CLIP score trung bình)
    best_shot = shots[0][0]
    best_seg = shots[0][1]
    best_avg_score = -1.0

    for shot, seg in shots:
        if frame_embeddings is not None and query_embedding is not None:
            validator = ShotValidator(min_frames=1, min_clip_score=0.0)
            shot = validator.validate(shot, query_embedding, frame_embeddings, frame_records)
        avg = shot.clip_score
        if avg > best_avg_score:
            best_avg_score = avg
            best_shot = shot
            best_seg = seg

    pipeline_stages["shot_validation"] = {
        "validated": best_shot.validated,
        "clip_score": best_shot.clip_score,
        "temporal_score": best_shot.temporal_score,
        "validation_score": best_shot.validation_score,
    }

    # Agent
    pipeline_stages["agent"] = {"max_steps": 5, "backend": vqa_backend}
    try:
        agent = create_agent(backend=vqa_backend)
        answer = agent.answer(shot=best_shot, question=question or query)
    except Exception as exc:
        LOGGER.exception("Agent failed")
        answer = f"[Agent error: {exc}]"

    pipeline_stages["agent"]["answer"] = answer[:300]

    return {
        "answer": answer,
        "results": [r.to_dict() for r in clip_results[:10]],
        "pipeline": pipeline_stages,
        "reasoning": "VQA pipeline: CLIP → Temporal → Validation → Agent",
    }


def trake_search(
    experiment: Experiment,
    query: str,
    context: str = "",
    top_k: int = 20,
    reranker=None,
    reranker_top_k: int = 10,
) -> dict:
    """TRAKE pipeline: CLIP search → (rerank) → temporal expansion → N event segments.

    Args:
        experiment:     Active experiment.
        query:          Event description; may contain multi-event lines (E1:, E2:, ...).
        context:        Optional context prepended to each event query.
        top_k:          First-stage candidate pool size (Qdrant).
        reranker:       Optional cross-encoder reranker; ``None`` skips reranking.
        reranker_top_k: Candidates retained after reranking.

    Returns:
        Dict with keys ``events``, ``results``, ``pipeline``.
    """
    import re
    from collections import defaultdict

    # Parse multi-event format (e.g., lines starting with E1:, E2:, 1., etc.)
    lines = [line.strip() for line in query.split('\n') if line.strip()]
    event_pattern = re.compile(r'^(?:E\d+|Event\s*\d+|\d+)\s*[:.]\s*(.*)$', re.IGNORECASE)

    event_queries = []
    prefix_context = context.strip()

    for line in lines:
        match = event_pattern.match(line)
        if match:
            event_queries.append(match.group(1).strip())
        else:
            if not event_queries:
                prefix_context = (prefix_context + " " + line).strip()
            else:
                event_queries[-1] = (event_queries[-1] + " " + line).strip()

    # If no structured sub-queries were parsed, default to the whole query as a single event
    if not event_queries:
        event_queries = [query.strip()]

    validator = ShotValidator(min_frames=1, min_clip_score=0.0)

    # 1. Single Event Query: Preserve original behavior
    if len(event_queries) == 1:
        retrieval_text = f"{prefix_context} {event_queries[0]}".strip()
        data = _run_temporal_pipeline(experiment, retrieval_text, top_k)
        pipeline_stages = data.get("pipeline", {})
        clip_results = data.get("clip_results", [])
        shots = data.get("shots", [])

        events = []
        frame_embeddings = data.get("frame_embeddings")
        frame_records = data.get("frame_records", [])
        query_embedding = data.get("query_embedding")

        for shot, seg in shots:
            if frame_embeddings is not None and query_embedding is not None:
                shot = validator.validate(shot, query_embedding, frame_embeddings, frame_records)
            events.append(
                {
                    "video_id": shot.video_id,
                    "video_name": shot.video_name,
                    "frame_count": shot.frame_count,
                    "start_timestamp": shot.start_timestamp,
                    "end_timestamp": shot.end_timestamp,
                    "score": shot.clip_score,
                    "frame_paths": shot.frame_paths[:5],
                }
            )

        events.sort(key=lambda x: x["score"], reverse=True)
        return {
            "events": events,
            "results": [r.to_dict() for r in clip_results[:10]],
            "pipeline": pipeline_stages,
        }

    # 2. Multi-Event Sequence Query
    candidates = defaultdict(lambda: defaultdict(list))
    all_clip_results = []
    combined_pipeline_stages = {}

    for idx, eq in enumerate(event_queries):
        retrieval_text = f"{prefix_context} {eq}".strip()
        data = _run_temporal_pipeline(experiment, retrieval_text, top_k)

        clip_results = data.get("clip_results", [])
        all_clip_results.extend(clip_results)

        stages = data.get("pipeline", {})
        combined_pipeline_stages[f"event_{idx + 1}"] = stages

        shots = data.get("shots", [])
        frame_embeddings = data.get("frame_embeddings")
        frame_records = data.get("frame_records", [])
        query_embedding = data.get("query_embedding")

        for shot, seg in shots:
            if frame_embeddings is not None and query_embedding is not None:
                # Đặt min_clip_score hợp lý để chặn rác
                validator = ShotValidator(min_frames=1, min_clip_score=0.15)
                shot = validator.validate(shot, query_embedding, frame_embeddings, frame_records)
            candidates[shot.video_id][idx].append(shot)

    # Find the best video_id and events sequence using DP sequence search
    best_video_id = None
    best_video_score = -1.0
    best_video_events = []

    def find_best_sequence(event_map):
        m_events = len(event_queries)
        for i in range(m_events):
            event_map[i].sort(key=lambda s: s.start_timestamp if s.start_timestamp is not None else 0.0)

        memo = {}

        def solve(idx, prev_end_time):
            if idx == m_events:
                return 0, 0.0, []

            # Làm tròn thời gian để dùng làm dict key an toàn
            state = (idx, round(prev_end_time, 1))
            if state in memo:
                return memo[state]

            # Option 1: Skip event idx
            best_covered, best_score, best_path = solve(idx + 1, prev_end_time)
            best_path = [None] + best_path

            # Option 2: Try all candidates
            for shot in event_map[idx]:
                shot_start = shot.start_timestamp if shot.start_timestamp is not None else 0.0
                shot_end = shot.end_timestamp if shot.end_timestamp is not None else shot_start
                
                if shot_start > prev_end_time:
                    # Tính khoảng cách thời gian giữa 2 event (càng gần càng tốt)
                    gap_penalty = 0.0
                    if prev_end_time > 0:
                        gap_penalty = min((shot_start - prev_end_time) * 0.01, 0.5)

                    cov, score, path = solve(idx + 1, shot_end)
                    current_cov = 1 + cov
                    current_score = shot.validation_score + score - gap_penalty

                    if (current_cov > best_covered) or (current_cov == best_covered and current_score > best_score):
                        best_covered = current_cov
                        best_score = current_score
                        best_path = [shot] + path

            memo[state] = (best_covered, best_score, best_path)
            return memo[state]

        return solve(0, 0.0)

    for video_id, event_map in candidates.items():
        covered, score, path = find_best_sequence(event_map)
        composite_score = covered * 1000.0 + score
        if composite_score > best_video_score:
            best_video_score = composite_score
            best_video_id = video_id
            best_video_events = path

    events = []
    if best_video_id is not None:
        for shot in best_video_events:
            if shot is not None:
                if not shot.video_name and all_clip_results:
                    for r in all_clip_results:
                        if r.video_id == best_video_id:
                            shot.video_name = r.video_name or ""
                            break
                events.append(
                    {
                        "video_id": shot.video_id,
                        "video_name": shot.video_name,
                        "frame_count": shot.frame_count,
                        "start_timestamp": shot.start_timestamp,
                        "end_timestamp": shot.end_timestamp,
                        "score": shot.clip_score,
                        "frame_paths": shot.frame_paths[:5],
                    }
                )

    # Sort the events chronologically to respect the temporal ordering
    events.sort(key=lambda x: x["start_timestamp"] if x["start_timestamp"] is not None else 0.0)

    # Take unique clip results to show in results panel
    seen_ids = set()
    unique_results = []
    for r in all_clip_results:
        if r.frame_id not in seen_ids:
            seen_ids.add(r.frame_id)
            unique_results.append(r)

    # Sort results globally by score descending
    unique_results.sort(key=lambda r: r.score, reverse=True)

    return {
        "events": events,
        "results": [r.to_dict() for r in unique_results[:10]],
        "pipeline": combined_pipeline_stages,
    }
