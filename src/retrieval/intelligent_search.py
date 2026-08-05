"""Intelligent search — one query, three modalities, LLM-weighted fusion.

Ported from the AIC_2025 reference project's ``/search/intelligent`` route:
an LLM reads the raw query once and splits it into a visual prompt (for KIS),
on-screen-text keywords (for OCR), and spoken keywords (for ASR), along with a
weight per modality. Each enabled modality searches independently and the
results are fused with the same weighted SRRF used for multi-model KIS
(see ``retrieval/fusion.py``).

Unlike the reference implementation (GPT-4o), the weighting/splitting LLM
call here goes through ``LlmQueryProcessor``, which already talks to whatever
backend ``.env`` configures (local vLLM or OpenRouter) — no new LLM plumbing.
"""

from __future__ import annotations

from config.settings import Experiment
from core.types import SearchResult
from retrieval.fusion import srrf_fuse
from retrieval.hydrator import ResultHydrator
from retrieval.text_search import NearestFrameIndex
from retrieval.vqa import _get_retriever
from stores.text.factory import build_text_index
from stores.vector.base import frame_result


def intelligent_search(
    experiment: Experiment,
    query: str,
    top_k: int = 20,
    enable_kis: bool = True,
    enable_ocr: bool = True,
    enable_asr: bool = True,
) -> dict:
    """Analyze ``query`` and search KIS/OCR/ASR in whatever mix the LLM picks.

    Returns a dict with ``results`` (fused, hydrated), ``analysis`` (the raw
    visual prompt / keywords / weights the query was split into), and
    ``component_counts`` (how many hits each modality contributed before
    fusion, for debugging why a result did or didn't show up).
    """
    query = query.strip()
    if not query:
        return {"results": [], "total": 0, "analysis": None, "component_counts": {}}

    retriever = _get_retriever(experiment)
    processed = retriever.query_processor.process(query)
    weights = dict(processed.weights)

    # Respect the caller's on/off switches by zeroing a disabled modality's
    # weight rather than skipping it outright — a component with real hits but
    # zero weight still shows up in component_counts for transparency.
    if not enable_kis:
        weights["kis"] = 0.0
    if not enable_ocr:
        weights["ocr"] = 0.0
    if not enable_asr:
        weights["asr"] = 0.0

    pool_size = max(top_k, 100)
    results_by_component: dict[str, list[SearchResult]] = {}
    component_counts: dict[str, int] = {}

    if enable_kis and weights.get("kis", 0) > 0:
        kis_hits = _search_kis(retriever, processed.visual_prompt, pool_size)
        results_by_component["kis"] = kis_hits
        component_counts["kis"] = len(kis_hits)

    if enable_ocr and weights.get("ocr", 0) > 0 and processed.ocr_keywords:
        ocr_hits = _search_text(experiment, processed.ocr_keywords, source="ocr", top_k=pool_size)
        results_by_component["ocr"] = ocr_hits
        component_counts["ocr"] = len(ocr_hits)

    if enable_asr and weights.get("asr", 0) > 0 and processed.asr_keywords:
        asr_hits = _search_text(experiment, processed.asr_keywords, source="asr", top_k=pool_size)
        results_by_component["asr"] = asr_hits
        component_counts["asr"] = len(asr_hits)

    if not results_by_component:
        return {
            "results": [],
            "total": 0,
            "analysis": _analysis_payload(processed),
            "component_counts": component_counts,
        }

    fused = srrf_fuse(results_by_component, top_k=pool_size, weights=weights)
    hydrated = ResultHydrator(experiment).hydrate(fused[:top_k])

    return {
        "results": [_result_payload(r) for r in hydrated],
        "total": len(hydrated),
        "analysis": _analysis_payload(processed),
        "component_counts": component_counts,
    }


def _search_kis(retriever, visual_prompt: str, top_k: int) -> list[SearchResult]:
    """Run the retriever's own embedding search (may itself be multi-model)."""
    return [
        SearchResult(
            frame_id=r.frame_id,
            video_id=r.video_id,
            score=r.score,
            frame_path=r.frame_path,
            video_path=r.video_path,
            video_name=r.video_name,
            shot_id=r.shot_id,
            frame_index=r.frame_index,
            timestamp_sec=r.timestamp_sec,
        )
        for r in retriever.search(visual_prompt, top_k=top_k)
    ]


def _search_text(experiment: Experiment, keywords: list[str], source: str, top_k: int) -> list[SearchResult]:
    """BM25 search each keyword, keeping the highest-scoring hit per frame.

    OCR documents carry a real ``frame_id``. ASR documents don't — speech
    isn't tied to one frame — so ASR hits are mapped to the nearest frame in
    the same video by timestamp (``NearestFrameIndex``, shared with the
    standalone ``/api/asr-search`` route).
    """
    index = build_text_index(experiment)
    nearest_frame = NearestFrameIndex(experiment) if source == "asr" else None

    best_score: dict[str, float] = {}
    for keyword in keywords:
        for doc in index.search_documents(keyword, top_k=top_k, source=source):
            frame_id = doc.get("frame_id")
            if not frame_id and nearest_frame is not None:
                frame_id = nearest_frame.nearest(doc.get("video_id", ""), doc.get("timestamp_sec") or 0.0)
            if not frame_id:
                continue
            score = float(doc.get("score", 0.0))
            if score > best_score.get(frame_id, -1.0):
                best_score[frame_id] = score

    ranked = sorted(best_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [frame_result(frame_id, score) for frame_id, score in ranked]


def _analysis_payload(processed) -> dict:
    return {
        "visual_prompt": processed.visual_prompt,
        "ocr_keywords": processed.ocr_keywords,
        "asr_keywords": processed.asr_keywords,
        "metadata": processed.metadata,
        "weights": processed.weights,
    }


def _result_payload(result: SearchResult) -> dict:
    return {
        "frame_id": result.frame_id,
        "video_id": result.video_id,
        "video_name": result.video_name or result.video_id,
        "frame_path": result.frame_path,
        "frame_index": result.frame_index,
        "shot_id": result.shot_id,
        "timestamp_sec": result.timestamp_sec,
        "score": round(result.score, 4),
    }
