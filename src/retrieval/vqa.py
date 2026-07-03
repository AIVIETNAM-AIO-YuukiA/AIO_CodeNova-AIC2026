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


def trake_search(
    experiment: Experiment,
    events: list[str],
    top_k: int = 300,
    window: int = 300,
) -> dict:
    """TRAKE pipeline: Bidirectional Pair Join (BPJ).

    Delegates to :mod:`retrieval.trake_search` — the core BPJ algorithm.
    See that module for full documentation.
    """
    from retrieval.trake_search import trake_search as _bpj_search

    return _bpj_search(
        experiment=experiment,
        events=events,
        top_k=top_k,
        window=window,
    )
