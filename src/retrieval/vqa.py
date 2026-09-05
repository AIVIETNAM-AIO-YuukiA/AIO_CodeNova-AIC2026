"""Pipeline orchestrator — kết nối Embedding search + Temporal Search + Agent.

4 pipeline theo tài liệu training:
  - Textual KIS: Query → Search → Temporal Search → Reranking → Submission
  - Video KIS:   (chưa implement)
  - VQA:         Query → Search → Temporal Search → Reranking → Shot Validation → Agent → Answer
  - TRAKE:       Query → Search → Temporal Search → Reranking → N events → Submission
"""

from __future__ import annotations

import logging
from threading import Lock

import numpy as np

from config.settings import Experiment
from modules.reranker.base import Reranker
from retrieval import build_retriever
from retrieval.temporal_search import (
    ShotValidator,
    find_segments,
    gather_frame_s,
    load_temporal_data,
)
from agent import create_agent

LOGGER = logging.getLogger(__name__)

# Retriever holds one embedder per configured model plus (for multi-model runs)
# the BLIP-2 reranker — several GB of VRAM. Rebuilding it per request, and per
# event within a multi-event TRAKE query, exhausts a small GPU within a couple
# of searches, so keep one instance per experiment.
_RETRIEVER_CACHE: dict[str, object] = {}
_RETRIEVER_CACHE_LOCK = Lock()


def _retriever_cache_key(experiment: Experiment) -> str:
    """Identify the concrete persisted run, not only its display name."""
    try:
        config_hash = experiment.config.config_hash()
    except (AttributeError, TypeError):
        config_hash = "unknown"
    return f"{experiment.run_dir.resolve()}::{config_hash}"


def _get_retriever(experiment: Experiment):
    """Return a cached retriever for ``experiment``, building it on first use."""
    cache_key = _retriever_cache_key(experiment)
    cached = _RETRIEVER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # UI requests are threaded. Building the same multi-GB retriever twice can
    # exhaust VRAM, so serialize only the cold initialization path.
    with _RETRIEVER_CACHE_LOCK:
        cached = _RETRIEVER_CACHE.get(cache_key)
        if cached is None:
            cached = build_retriever(experiment)
            _RETRIEVER_CACHE[cache_key] = cached
    return cached


def _run_temporal_pipeline(
    experiment: Experiment,
    query: str,
    top_k: int = 20,
    reranker: Reranker | None = None,
    reranker_top_k: int = 10,
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
    use_llm: bool = True,
) -> dict:
    """Run embedding search, optional reranking, and temporal expansion.

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
    retriever = _get_retriever(experiment)
    embed_results = retriever.search(
        query=query,
        top_k=top_k,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
    )
    pipeline_stages["embed_search"] = {
        "top_k": top_k,
        "results_count": len(embed_results),
    }

    # Stage 2 (optional): cross-encoder reranking over the first-stage pool.
    if reranker is not None and embed_results:
        LOGGER.info("Reranker: scoring %d candidates...", len(embed_results))
        embed_results = reranker.rerank(query=query, results=embed_results)
        embed_results = embed_results[:reranker_top_k]
        pipeline_stages["rerank"] = {
            "model": getattr(reranker, "model_name", "unknown"),
            "reranker_top_k": reranker_top_k,
            "results_after_rerank": len(embed_results),
        }
        LOGGER.info("Reranker: kept %d results after reranking.", len(embed_results))

    if not embed_results:
        return {
            "pipeline": pipeline_stages,
            "embed_results": [],
            "segments": [],
            "frame_embeddings": None,
            "frame_records": [],
            "query_embedding": None,
        }

    # Load temporal data
    try:
        frame_embeddings, frame_records = load_temporal_data(experiment)
    except FileNotFoundError as exc:
        pipeline_stages["error"] = str(exc)
        return {
            "pipeline": pipeline_stages,
            "embed_results": embed_results,
            "segments": [],
            "frame_embeddings": None,
            "frame_records": [],
            "query_embedding": None,
        }

    # Map embedding search results to positions in sorted embeddings
    hit_positions = set()
    for r in embed_results:
        for i, rec in enumerate(frame_records):
            if rec.get("frame_id") == r.frame_id:
                hit_positions.add(i)
                break

    if not hit_positions:
        pipeline_stages["temporal_search"] = {"error": "No matching positions found"}
        return {
            "pipeline": pipeline_stages,
            "embed_results": embed_results,
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

    # Query embedding cho shot validation — cùng model với load_temporal_data.
    # Reuse the cached retriever's embedder instead of building a fresh one:
    # this runs once per event in a multi-event TRAKE query, and a second
    # full model load per event exhausts a small GPU within a few searches.
    embedder = _get_retriever(experiment).embedders[experiment.config.embedding_models[0]]
    query_embedding = embedder.embed_text(query)

    # Gather shot cho mỗi segment
    video_names = {
        result.video_id: result.video_name
        for result in embed_results
        if result.video_name
    }
    shots = []
    for seg in segments:
        shot = gather_frame_s(seg, frame_records)
        if shot and shot.frame_count > 0:
            shot.video_name = video_names.get(shot.video_id, shot.video_name or shot.video_id)
            shots.append((shot, seg))

    pipeline_stages["gather_shot"] = {
        "shots_count": len(shots),
    }

    return {
        "pipeline": pipeline_stages,
        "embed_results": embed_results,
        "segments": segments,
        "shots": shots,
        "frame_embeddings": frame_embeddings,
        "frame_records": frame_records,
        "query_embedding": np.asarray(query_embedding),
    }


def _cached_captions_context(experiment: Experiment, shot) -> str:
    """Best-effort: captions cached at index time for the shot's frames.

    Empty string when the experiment never ran captioning — the agent then
    relies on its caption/ocr tools alone.
    """
    try:
        from repository.caption_repo import CaptionRepository

        captions = CaptionRepository(experiment).by_id()
    except Exception:
        LOGGER.debug("Caption manifest unavailable", exc_info=True)
        return ""
    lines = []
    for frame in shot.frames[:5]:
        caption = captions.get(str(frame.get("frame_id")))
        if caption:
            lines.append(f"- {caption}")
    return "\n".join(lines)


def _legacy_vqa_search(
    experiment: Experiment,
    query: str,
    question: str,
    context: str = "",
    top_k: int = 20,
    reranker=None,
    reranker_top_k: int = 10,
    vqa_backend: str = "local",
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
    use_llm: bool = True,
) -> dict:
    """VQA pipeline: Embedding search → (rerank) → temporal → shot validation → agent answer.

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
    data = _run_temporal_pipeline(
        experiment,
        retrieval_text,
        top_k,
        reranker=reranker,
        reranker_top_k=reranker_top_k,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
    )
    pipeline_stages = data.get("pipeline", {})
    embed_results = data.get("embed_results", [])
    shots = data.get("shots", [])
    frame_embeddings = data.get("frame_embeddings")
    frame_records = data.get("frame_records", [])
    query_embedding = data.get("query_embedding")

    if not embed_results:
        return {
            "answer": "No relevant frames found.",
            "results": [],
            "pipeline": pipeline_stages,
            "reasoning": "",
        }

    shots = data.get("shots", [])
    if not shots:
        # Fallback: dùng embedding search result đầu tiên
        return {
            "answer": "Could not find a valid shot segment.",
            "results": [r.to_dict() for r in embed_results[:5]],
            "pipeline": pipeline_stages,
            "reasoning": "Temporal search found no segments.",
        }

    # Shot validation: chọn shot tốt nhất (based on embedding score trung bình)
    best_shot = shots[0][0]
    shots[0][1]
    best_avg_score = -1.0

    for shot, seg in shots:
        if frame_embeddings is not None and query_embedding is not None:
            validator = ShotValidator(min_frames=1, min_sim_score=0.0)
            shot = validator.validate(shot, query_embedding, frame_embeddings, frame_records)
        avg = shot.sim_score
        if avg > best_avg_score:
            best_avg_score = avg
            best_shot = shot

    pipeline_stages["shot_validation"] = {
        "validated": best_shot.validated,
        "sim_score": best_shot.sim_score,
        "temporal_score": best_shot.temporal_score,
        "validation_score": best_shot.validation_score,
    }

    # Agent — kèm caption đã cache lúc index (nếu có) để LLM text-only vẫn có
    # ngữ cảnh thị giác khi VLM caption/ocr service không chạy.
    pipeline_stages["agent"] = {"max_steps": 5, "backend": vqa_backend}
    agent_error = None
    try:
        agent = create_agent(backend=vqa_backend)
        answer = agent.answer(
            shot=best_shot,
            question=question or query,
            context=_cached_captions_context(experiment, best_shot),
        )
    except Exception as exc:
        LOGGER.exception("Agent failed")
        agent_error = str(exc)
        answer = ""

    pipeline_stages["agent"]["answer"] = answer[:300] if answer else ""

    result: dict = {
        "answer": answer,
        "results": [r.to_dict() for r in embed_results[:10]],
        "pipeline": pipeline_stages,
        "reasoning": "VQA pipeline: Embedding search → Temporal → Validation → Agent",
    }
    if agent_error:
        result["agent_error"] = agent_error
    return result


def vqa_search(
    experiment: Experiment,
    query: str,
    question: str,
    context: str = "",
    top_k: int = 20,
    reranker=None,
    reranker_top_k: int = 10,
    vqa_backend: str = "local",
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
    use_llm: bool = True,
    pipeline_mode: str = "grounded",
) -> dict:
    """Run grounded multi-frame VQA, with an explicit legacy rollback mode.

    ``top_k`` controls only how many retrieval cards are returned by the
    grounded pipeline. Candidate recall uses a fixed internal pool so changing
    the UI display count does not silently change which evidence answers the
    question.
    """
    normalized_mode = pipeline_mode.strip().lower()
    if normalized_mode not in {"grounded", "legacy"}:
        raise ValueError("pipeline_mode must be 'grounded' or 'legacy'")
    if normalized_mode == "legacy":
        result = _legacy_vqa_search(
            experiment=experiment,
            query=query,
            question=question,
            context=context,
            top_k=top_k,
            reranker=reranker,
            reranker_top_k=reranker_top_k,
            vqa_backend=vqa_backend,
            enabled_models=enabled_models,
            use_reranker=use_reranker,
            use_llm=use_llm,
        )
        result["pipeline_mode"] = "legacy"
        return result

    # Imported lazily to keep the existing retrieval import graph free of a
    # vqa -> grounded_vqa -> text_search -> retrieval cycle during startup.
    from retrieval.grounded_vqa import grounded_vqa_search

    result = grounded_vqa_search(
        experiment=experiment,
        retriever=_get_retriever(experiment),
        query=query,
        question=question,
        context=context,
        top_k=top_k,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
    )
    result["pipeline_mode"] = "grounded"
    return result


def trake_search(
    experiment: Experiment,
    query: str,
    context: str = "",
    top_k: int = 300,
    reranker=None,
    reranker_top_k: int = 10,
) -> dict:
    """TRAKE pipeline: delegates to BPJ algorithm for multi-event queries."""
    import re
    from retrieval.trake_search import trake_bpj_search

    # Parse multi-event format (e.g., lines starting with E1:, E2:, 1., etc.)
    lines = [line.strip() for line in query.split("\n") if line.strip()]
    event_pattern = re.compile(r"^(?:E\d+|Event\s*\d+|\d+)\s*[:.]\s*(.*)$", re.IGNORECASE)

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

    if not event_queries:
        event_queries = [query.strip()]

    if prefix_context:
        event_queries = [f"{prefix_context} {eq}".strip() for eq in event_queries]

    if len(event_queries) < 2:
        # Fallback for single event query
        data = _run_temporal_pipeline(experiment, event_queries[0], top_k)
        pipeline_stages = data.get("pipeline", {})
        embed_results = data.get("embed_results", [])
        shots = data.get("shots", [])

        validator = ShotValidator(min_frames=1, min_sim_score=0.0)
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
                    "score": shot.sim_score,
                    "frame_paths": shot.frame_paths[:5],
                }
            )

        events.sort(key=lambda x: x["score"], reverse=True)
        return {
            "events": events,
            "results": [r.to_dict() for r in embed_results[:10]],
            "pipeline": pipeline_stages,
        }

    return trake_bpj_search(
        experiment=experiment,
        events=event_queries,
        top_k=top_k,
        window=15,
    )


def enhanced_temporal_search(
    experiment: Experiment,
    query: str,
    context: str = "",
    top_k: int = 20,
    max_events: int = 5,
    reranker=None,
    reranker_top_k: int = 10,
) -> dict:
    """TRAKE, but the event list is extracted from one sentence by an LLM.

    Ported from the AIC_2025 reference project's ``/search/temporal/enhanced``:
    the user types a single natural-language description of a sequence
    ("a man walks out, then rides away") instead of filling in each event box,
    and the LLM splits it into ordered events. Everything after that — per-event
    retrieval, temporal expansion, sequence search — is exactly ``trake_search``.

    Returns ``trake_search``'s payload plus ``extracted_events`` so the UI can
    show what the query was split into. Falls back to searching the query as a
    single event when the LLM is unavailable or finds no sequence.
    """
    retriever = _get_retriever(experiment)
    events = retriever.query_processor.extract_temporal_events(query, max_events=max_events)

    if len(events) >= 2:
        trake_query = "\n".join(f"E{i}: {event}" for i, event in enumerate(events, start=1))
    else:
        # No sequence detected: let trake_search treat the whole query as one event.
        trake_query = query

    result = trake_search(
        experiment=experiment,
        query=trake_query,
        context=context,
        top_k=top_k,
        reranker=reranker,
        reranker_top_k=reranker_top_k,
    )
    result["extracted_events"] = events
    return result
