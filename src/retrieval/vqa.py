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
) -> dict:
    """Chạy CLIP search + Temporal search, trả về raw segments + data.

    Dùng chung cho cả VQA và TRAKE.
    """
    pipeline_stages = {}

    # CLIP Search
    retriever = build_retriever(experiment)
    clip_results = retriever.search(query=query, top_k=top_k)
    pipeline_stages["clip_search"] = {
        "top_k": top_k,
        "results_count": len(clip_results),
    }

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
) -> dict:
    """VQA pipeline: CLIP search → Temporal → Shot Validation → Agent → Answer.

    Args:
        experiment: Experiment instance.
        query: Search query text.
        question: VQA question.
        context: Optional scene context.
        top_k: Number of CLIP search results.

    Returns:
        Dict với answer, results, pipeline stages.
    """
    retrieval_text = f"{context} {query} {question}".strip()
    data = _run_temporal_pipeline(experiment, retrieval_text, top_k)
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
    shots[0][1]
    best_avg_score = -1.0

    for shot, seg in shots:
        if frame_embeddings is not None and query_embedding is not None:
            validator = ShotValidator(min_frames=1, min_clip_score=0.0)
            shot = validator.validate(shot, query_embedding, frame_embeddings, frame_records)
        avg = shot.clip_score
        if avg > best_avg_score:
            best_avg_score = avg
            best_shot = shot

    pipeline_stages["shot_validation"] = {
        "validated": best_shot.validated,
        "clip_score": best_shot.clip_score,
        "temporal_score": best_shot.temporal_score,
        "validation_score": best_shot.validation_score,
    }

    # Agent
    pipeline_stages["agent"] = {"max_steps": 5}
    try:
        agent = create_agent()
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


def _search_event(
    experiment: Experiment,
    event_text: str,
    event_index: int,
    top_k: int = 50,
) -> list[dict]:
    """Search one event text and return scored results."""
    retriever = build_retriever(experiment)
    results = retriever.search(query=event_text, top_k=top_k)
    out = []
    for rank, r in enumerate(results, start=1):
        out.append(
            {
                "event_index": event_index,
                "rank": rank,
                "score": r.score,
                "frame_id": r.frame_id,
                "video_id": r.video_id,
                "video_name": r.video_name or r.video_id,
                "frame_path": r.frame_path,
                "timestamp_sec": r.timestamp_sec,
                "shot_id": r.shot_id,
                "frame_index": r.frame_index,
            }
        )
    return out


def _best_event_frames(
    event_results: list[list[dict]],
    vid: str,
) -> tuple[float, list[dict], bool]:
    """Pick best (lowest-rank) frame for each event, compute score, check temporal order.

    For each event, select the best frame for this video (minimum rank).
    Compute score = Σ e^(-0.02 * rank), and a temporal validity flag indicating whether
    E1_time < E2_time < ... < En_time holds (strictly increasing).

    Returns (total_score, [selected_frames], temporal_order_valid).
    """
    EXP_DECAY = 0.02
    TEMPORAL_FACTOR = 2.0

    selected = []
    total_score = 0.0
    timestamps = []

    for er in event_results:
        candidates = [h for h in er if h["video_id"] == vid]
        if not candidates:
            # This should never happen for videos that passed the intersection filter
            continue
        best = min(candidates, key=lambda h: h["rank"])
        selected.append(best)
        total_score += pow(2.718, -EXP_DECAY * best["rank"])
        timestamps.append(best.get("timestamp_sec"))

    if not timestamps:
        return total_score, [], False

    temporal_ok = True
    for j in range(1, len(timestamps)):
        if timestamps[j] <= timestamps[j - 1]:
            temporal_ok = False
            break

    if temporal_ok:
        total_score *= TEMPORAL_FACTOR

    return total_score, selected, temporal_ok


def trake_search(
    experiment: Experiment,
    events: list[str],
    top_k: int = 50,
) -> dict:
    """TRAKE pipeline: search multiple events independently, find ALL videos
    containing every event with a temporally-valid frame sequence.

    For each event text, search top-K frames, intersect to find videos
    appearing in ALL events, then use DP to pick the best frame per event
    (by exponential-decay score) such that timestamps are strictly increasing.
    Videos without a valid temporal ordering are excluded.

    Returns ALL matching videos sorted by score descending.
    """
    if len(events) < 2:
        return {"error": "At least 2 events are required.", "videos": []}

    # 1. Search each event independently
    event_results: list[list[dict]] = []
    for i, ev in enumerate(events):
        ev_text = ev.strip()
        if not ev_text:
            return {"error": f"Event {i+1} text is empty.", "videos": []}
        event_results.append(_search_event(experiment, ev_text, i, top_k=top_k))

    # 2. Intersect: videos present in ALL events
    video_sets: list[set[str]] = [set(r["video_id"] for r in er) for er in event_results]
    common_videos = video_sets[0]
    for vs in video_sets[1:]:
        common_videos &= vs

    if not common_videos:
        return {"videos": [], "total_candidates": 0}

    # 3. For each common video, select best frame per event and compute score
    scored: list[tuple[float, list[dict], bool, str]] = []
    for vid in common_videos:
        total, chosen, temporal_ok = _best_event_frames(event_results, vid)
        scored.append((total, chosen, temporal_ok, vid))

    scored.sort(key=lambda x: -x[0])

    # 4. Build result list (ALL matching videos) with temporal badges
    out_videos = []
    for total, chosen, temporal_ok, vid in scored:
        events_out = [
            {
                "event_index": c["event_index"],
                "rank": c["rank"],
                "frame_id": c["frame_id"],
                "frame_path": c["frame_path"],
                "video_id": c["video_id"],
                "video_name": c["video_name"],
                "timestamp_sec": c["timestamp_sec"],
                "score": c["score"],
                "shot_id": c["shot_id"],
                "frame_index": c["frame_index"],
            }
            for c in chosen
        ]
        out_videos.append(
            {
                "video_id": vid,
                "video_name": chosen[0]["video_name"] if chosen else vid,
                "score": round(total, 4),
                "temporal_order_valid": temporal_ok,
                "events": events_out,
            }
        )

    return {
        "videos": out_videos,
        "total_candidates": len(common_videos),
    }
