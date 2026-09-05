"""Adaptive multimodal Intelligent Search over visual, OCR and ASR."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from statistics import median
from time import perf_counter

from config.settings import Experiment
from core.logging import get_logger
from core.types import SearchResult
from retrieval.fusion import srrf_fuse
from retrieval.hydrator import ResultHydrator
from retrieval.text_search import AsrTemporalMapper, NearestFrameIndex
from retrieval.vqa import _get_retriever
from stores.text.factory import build_text_index
from stores.vector.base import frame_result

_COMPONENTS = ("kis", "ocr", "asr")
LOGGER = get_logger(__name__)


@dataclass
class TextSearchOutcome:
    results: list[SearchResult]
    evidence_by_frame: dict[str, list[dict]]
    stats: dict[str, object]


def intelligent_search(
    experiment: Experiment,
    query: str,
    top_k: int = 20,
    enable_kis: bool = True,
    enable_ocr: bool = True,
    enable_asr: bool = True,
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
    use_llm: bool | None = True,
    fusion_mode: str = "adaptive",
    text_search_mode: str = "separate",
    temporal_asr: bool = True,
    use_evidence_reranker: bool | None = None,
    max_frames_per_shot: int = 2,
) -> dict:
    """Search all routed modalities and return results plus diagnostics.

    The query is decomposed exactly once. ``fixed`` fusion is retained for
    evaluation, while ``adaptive`` derives auxiliary weights from router
    confidence and observed hit quality. ``joint`` text search is an
    experimental OCR+ASR ablation and is never mixed with the separate lists.
    """
    started = perf_counter()
    query = query.strip()
    if fusion_mode not in {"adaptive", "fixed"}:
        raise ValueError("fusion_mode must be 'adaptive' or 'fixed'")
    if text_search_mode not in {"separate", "joint"}:
        raise ValueError("text_search_mode must be 'separate' or 'joint'")
    if not query:
        return _empty_response()

    timing_ms: dict[str, float] = {}
    retriever = _get_retriever(experiment)
    tick = perf_counter()
    processed = retriever.query_processor.process(
        query,
        enabled_models=enabled_models,
        use_llm=use_llm is not False,
    )
    timing_ms["query_processing"] = _elapsed_ms(tick)
    confidences = _modality_confidences(processed)

    enabled = {
        "kis": enable_kis,
        "ocr": enable_ocr,
        "asr": enable_asr,
    }
    component_status = {
        name: "skipped" if is_enabled else "disabled" for name, is_enabled in enabled.items()
    }
    component_counts = {name: 0 for name in _COMPONENTS}
    results_by_component: dict[str, list[SearchResult]] = {}
    component_stats: dict[str, dict] = {}
    evidence_by_frame: dict[str, list[dict]] = {}
    pool_size = max(top_k, 100)

    if enable_kis:
        tick = perf_counter()
        kis_results = _search_kis(
            retriever,
            processed,
            top_k=pool_size,
            enabled_models=enabled_models,
            use_reranker=use_reranker,
        )
        timing_ms["kis"] = _elapsed_ms(tick)
        results_by_component["kis"] = kis_results
        component_counts["kis"] = len(kis_results)
        component_status["kis"] = "used" if kis_results else "no_hits"
        component_stats["kis"] = {
            "router_confidence": confidences["kis"],
            "score_separation": _score_separation([item.score for item in kis_results]),
            "effective_weight": 1.0,
            "hit_count": len(kis_results),
        }

    text_specs: list[tuple[str, list[str], str | tuple[str, ...], float]] = []
    if text_search_mode == "joint" and enable_ocr and enable_asr:
        joint_keywords = _unique(
            (*_keywords_for(processed, "ocr"), *_keywords_for(processed, "asr"))
        )
        if joint_keywords and max(confidences["ocr"], confidences["asr"]) > 0:
            text_specs.append(
                (
                    "joint_text",
                    joint_keywords,
                    ("ocr", "asr"),
                    max(confidences["ocr"], confidences["asr"]),
                )
            )
    else:
        if enable_ocr and processed.ocr_keywords and confidences["ocr"] > 0:
            text_specs.append(("ocr", _keywords_for(processed, "ocr"), "ocr", confidences["ocr"]))
        if enable_asr and processed.asr_keywords and confidences["asr"] > 0:
            text_specs.append(("asr", _keywords_for(processed, "asr"), "asr", confidences["asr"]))

    fusion_weights: dict[str, float] = {"kis": 1.0 if enable_kis else 0.0}
    for component, keywords, sources, confidence in text_specs:
        tick = perf_counter()
        try:
            outcome = _search_text(
                experiment,
                keywords,
                source=sources,
                top_k=pool_size,
                temporal_asr=temporal_asr,
            )
        except Exception as exc:
            timing_ms[component] = _elapsed_ms(tick)
            fusion_weights[component] = 0.0
            component_stats[component] = {
                "router_confidence": confidence,
                "effective_weight": 0.0,
                "hit_count": 0,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
            if component == "joint_text":
                component_counts["joint_text"] = 0
                component_status["ocr"] = "error"
                component_status["asr"] = "error"
            else:
                component_status[component] = "error"
            LOGGER.exception(
                "event=INTELLIGENT_TEXT_COMPONENT_DEGRADED component=%s; "
                "preserving remaining modalities",
                component,
            )
            continue
        timing_ms[component] = _elapsed_ms(tick)
        weight = (
            _adaptive_weight(confidence, outcome.stats)
            if fusion_mode == "adaptive"
            else _fixed_weight(component)
        )
        outcome.stats.update({"router_confidence": confidence, "effective_weight": weight})
        component_stats[component] = outcome.stats
        fusion_weights[component] = weight

        if component == "joint_text":
            component_counts["joint_text"] = len(outcome.results)
            source_counts = outcome.stats.get("source_hit_counts", {})
            for source_name in ("ocr", "asr"):
                source_count = int(source_counts.get(source_name, 0))
                component_counts[source_name] = source_count
                component_status[source_name] = (
                    "used" if source_count > 0 and weight > 0 else "no_hits"
                )
        else:
            component_counts[component] = len(outcome.results)
            component_status[component] = "used" if outcome.results and weight > 0 else "no_hits"

        if outcome.results and weight > 0:
            results_by_component[component] = outcome.results
            for frame_id, records in outcome.evidence_by_frame.items():
                evidence_by_frame.setdefault(frame_id, []).extend(records)

    for component in ("ocr", "asr"):
        if enabled[component] and component_status[component] == "skipped":
            keywords = getattr(processed, f"{component}_keywords", [])
            component_stats.setdefault(
                component,
                {
                    "router_confidence": confidences[component],
                    "effective_weight": 0.0,
                    "hit_count": 0,
                    "skip_reason": (
                        "no_keywords_or_zero_confidence" if not keywords else "zero_confidence"
                    ),
                },
            )

    if not results_by_component:
        timing_ms["total"] = _elapsed_ms(started)
        return {
            "results": [],
            "total": 0,
            "analysis": _analysis_payload(processed),
            "component_counts": component_counts,
            "component_status": component_status,
            "component_stats": component_stats,
            "component_scores": {},
            "fusion_weights": fusion_weights,
            "fusion_mode": fusion_mode,
            "timing_ms": timing_ms,
        }

    component_scores = _component_score_diagnostics(results_by_component)
    tick = perf_counter()
    fused = srrf_fuse(results_by_component, top_k=pool_size, weights=fusion_weights)
    fused = _normalize_results(fused)
    timing_ms["fusion"] = _elapsed_ms(tick)

    tick = perf_counter()
    hydrated = ResultHydrator(experiment).hydrate(fused)
    apply_evidence_reranker = (
        use_evidence_reranker if use_evidence_reranker is not None else use_reranker is not False
    )
    if apply_evidence_reranker:
        try:
            hydrated = _evidence_rerank(
                hydrated,
                component_scores,
                fusion_weights,
                evidence_by_frame,
                limit=30,
            )
        except Exception:  # pragma: no cover - defensive fail-open boundary
            LOGGER.exception("event=EVIDENCE_RERANKER_DEGRADED; returning pre-rerank SRRF results")
    hydrated = _diversify_by_shot(hydrated, top_k=top_k, limit=max_frames_per_shot)
    timing_ms["rerank_and_diversify"] = _elapsed_ms(tick)
    timing_ms["total"] = _elapsed_ms(started)

    returned_component_scores = {
        result.frame_id: component_scores.get(result.frame_id, {}) for result in hydrated
    }
    return {
        "results": [
            _result_payload(
                result,
                component_scores=returned_component_scores.get(result.frame_id, {}),
                evidence=evidence_by_frame.get(result.frame_id, []),
            )
            for result in hydrated
        ],
        "total": len(hydrated),
        "analysis": _analysis_payload(processed),
        "component_counts": component_counts,
        "component_status": component_status,
        "component_stats": component_stats,
        "component_scores": returned_component_scores,
        "fusion_weights": fusion_weights,
        "fusion_mode": fusion_mode,
        "timing_ms": timing_ms,
    }


def _search_kis(
    retriever,
    processed,
    top_k: int,
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
) -> list[SearchResult]:
    """Run visual retrieval without processing the query a second time."""
    return [
        SearchResult(
            frame_id=result.frame_id,
            video_id=result.video_id,
            score=result.score,
            frame_path=result.frame_path,
            video_path=result.video_path,
            video_name=result.video_name,
            shot_id=result.shot_id,
            frame_index=result.frame_index,
            timestamp_sec=result.timestamp_sec,
            caption=result.caption,
        )
        for result in retriever.search_processed(
            processed,
            top_k=top_k,
            enabled_models=enabled_models,
            use_reranker=use_reranker,
        )
    ]


def _search_text(
    experiment: Experiment,
    keywords: list[str],
    source: str | tuple[str, ...],
    top_k: int,
    temporal_asr: bool = True,
) -> TextSearchOutcome:
    """Search BM25 phrases and aggregate coverage/quality-aware frame scores."""
    index = build_text_index(experiment)
    sources = (source,) if isinstance(source, str) else tuple(source)
    nearest = NearestFrameIndex(experiment) if "asr" in sources or "ocr" in sources else None
    temporal = AsrTemporalMapper(experiment) if temporal_asr and "asr" in sources else None
    unique_keywords = _unique(keywords)
    semantic_keywords = {_normalize_text(keyword) for keyword in unique_keywords}
    per_frame: dict[str, dict] = {}
    hit_keywords: set[str] = set()
    hit_semantic_keywords: set[str] = set()
    source_hit_frames: dict[str, set[str]] = {name: set() for name in sources}
    seen_ocr: dict[tuple[str, str, str, str], list[float]] = {}

    for keyword in unique_keywords:
        semantic_keyword = _normalize_text(keyword)
        documents = index.search_documents(keyword, top_k=top_k, source=source)
        raw_scores = [float(document.get("score", 0.0)) for document in documents]
        minimum = min(raw_scores, default=0.0)
        maximum = max(raw_scores, default=0.0)
        span = maximum - minimum
        for document in documents:
            document_source = str(document.get("source") or sources[0])
            text = str(document.get("text") or "").strip()
            if not text:
                continue
            quality = _document_quality(text, document_source)
            if quality <= 0:
                continue
            raw_score = float(document.get("score", 0.0))
            normalized_score = (raw_score - minimum) / span if span > 1e-9 else 1.0
            exact_phrase = _normalize_text(keyword) in _normalize_text(text)

            if document_source == "ocr":
                frame_id = str(document.get("frame_id") or "")
                frame_context = nearest.frame_context(frame_id) if nearest is not None else None
                shot_id = frame_context[0] if frame_context else frame_id
                raw_timestamp = document.get("timestamp_sec")
                if raw_timestamp is None and frame_context:
                    raw_timestamp = frame_context[1]
                timestamp = float(raw_timestamp or 0.0)
                dedupe_key = (
                    semantic_keyword,
                    str(document.get("video_id") or ""),
                    shot_id,
                    _normalize_text(text),
                )
                previous_timestamps = seen_ocr.setdefault(dedupe_key, [])
                if any(abs(timestamp - previous) <= 2.0 for previous in previous_timestamps):
                    continue
                previous_timestamps.append(timestamp)

            mapped_frames = _map_document_frames(
                document,
                document_source,
                nearest=nearest,
                temporal=temporal,
                temporal_asr=temporal_asr,
            )
            if not mapped_frames:
                continue
            segment_interval = (
                temporal.interval_for(document)
                if document_source == "asr" and temporal is not None
                else None
            )
            hit_keywords.add(keyword)
            hit_semantic_keywords.add(semantic_keyword)
            for frame_id, temporal_weight in mapped_frames:
                source_hit_frames.setdefault(document_source, set()).add(frame_id)
                contribution = normalized_score * quality * temporal_weight
                record = per_frame.setdefault(
                    frame_id,
                    {"scores": [], "keywords": set(), "qualities": [], "evidence": []},
                )
                record["scores"].append(contribution)
                record["keywords"].add(semantic_keyword)
                record["qualities"].append(quality)
                if len(record["evidence"]) < 8:
                    evidence_record = {
                        "source": document_source,
                        "doc_id": document.get("doc_id"),
                        "text": text[:500],
                        "keyword": keyword,
                        "exact_phrase": exact_phrase,
                        "document_quality": round(quality, 4),
                        "temporal_weight": round(temporal_weight, 4),
                        "bm25_score": round(raw_score, 4),
                    }
                    if segment_interval is not None:
                        evidence_record["segment_start_sec"] = round(segment_interval[0], 4)
                        evidence_record["segment_end_sec"] = round(segment_interval[1], 4)
                    record["evidence"].append(evidence_record)

    ranked_rows: list[tuple[str, float, float, list[dict]]] = []
    for frame_id, record in per_frame.items():
        matched = len(record["keywords"])
        score = max(record["scores"], default=0.0) * (1.0 + 0.2 * max(0, matched - 1))
        avg_quality = sum(record["qualities"]) / len(record["qualities"])
        ranked_rows.append((frame_id, score, avg_quality, record["evidence"]))
    ranked_rows.sort(key=lambda row: row[1], reverse=True)
    ranked_rows = ranked_rows[:top_k]
    results = [frame_result(frame_id, score) for frame_id, score, _, _ in ranked_rows]
    evidence = {frame_id: records for frame_id, _, _, records in ranked_rows}
    keyword_coverage = (
        len(hit_semantic_keywords) / len(semantic_keywords) if semantic_keywords else 0.0
    )
    score_separation = _score_separation([row[1] for row in ranked_rows])
    document_quality = (
        sum(row[2] for row in ranked_rows[:10]) / min(len(ranked_rows), 10) if ranked_rows else 0.0
    )
    hit_quality = 0.5 * keyword_coverage + 0.3 * score_separation + 0.2 * document_quality
    return TextSearchOutcome(
        results=results,
        evidence_by_frame=evidence,
        stats={
            "hit_count": len(results),
            "keyword_count": len(semantic_keywords),
            "query_variant_count": len(unique_keywords),
            "matched_keywords": sorted(hit_keywords),
            "keyword_coverage": round(keyword_coverage, 6),
            "score_separation": round(score_separation, 6),
            "document_quality": round(document_quality, 6),
            "hit_quality": round(hit_quality, 6),
            "source_hit_counts": {
                name: len(frame_ids) for name, frame_ids in source_hit_frames.items()
            },
        },
    )


def _map_document_frames(
    document: dict,
    source: str,
    nearest: NearestFrameIndex | None,
    temporal: AsrTemporalMapper | None,
    temporal_asr: bool,
) -> list[tuple[str, float]]:
    frame_id = document.get("frame_id")
    if frame_id:
        return [(str(frame_id), 1.0)]
    if source != "asr" or nearest is None:
        return []
    if temporal_asr and temporal is not None:
        return temporal.map_document(document)
    nearest_id = nearest.nearest(
        str(document.get("video_id") or ""), float(document.get("timestamp_sec") or 0.0)
    )
    return [(nearest_id, 1.0)] if nearest_id else []


def _adaptive_weight(confidence: float, stats: dict) -> float:
    hit_quality = float(stats.get("hit_quality", 0.0))
    return round(min(0.5, max(0.0, confidence * hit_quality)), 6)


def _fixed_weight(component: str) -> float:
    """Return the query-independent auxiliary baseline used by ablations."""
    return 0.3 if component in {"ocr", "asr", "joint_text"} else 0.0


def _modality_confidences(processed) -> dict[str, float]:
    supplied = getattr(processed, "modality_confidence", {}) or {}
    result = {"kis": 1.0, "ocr": 0.0, "asr": 0.0}
    for component, default in result.items():
        try:
            result[component] = min(1.0, max(0.0, float(supplied.get(component, default))))
        except (TypeError, ValueError):
            pass
    for component in ("ocr", "asr"):
        keywords = getattr(processed, f"{component}_keywords", [])
        bonus = float(processed.weights.get(f"{component}_bonus", 0.0))
        if keywords and result[component] <= 0:
            result[component] = min(1.0, max(0.5, bonus * 2.0))
    return result


def _component_score_diagnostics(
    results_by_component: dict[str, list[SearchResult]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    payload: dict[str, dict[str, dict[str, float | int]]] = {}
    for component, results in results_by_component.items():
        if not results:
            continue
        scores = [result.score for result in results]
        minimum, maximum = min(scores), max(scores)
        span = maximum - minimum
        sorted_results = sorted(results, key=lambda item: item.score, reverse=True)
        for rank, result in enumerate(sorted_results, 1):
            normalized = (result.score - minimum) / span if span > 1e-9 else 1.0
            payload.setdefault(result.frame_id, {})[component] = {
                "rank": rank,
                "raw_score": round(float(result.score), 6),
                "normalized_score": round(float(normalized), 6),
            }
    return payload


def _evidence_rerank(
    results: list[SearchResult],
    component_scores: dict[str, dict[str, dict[str, float | int]]],
    fusion_weights: dict[str, float],
    evidence_by_frame: dict[str, list[dict]],
    limit: int,
) -> list[SearchResult]:
    if not any(evidence_by_frame.values()):
        return results
    reranked: list[tuple[int, SearchResult]] = []
    for original_rank, result in enumerate(results[:limit]):
        scores = component_scores.get(result.frame_id, {})
        evidence = evidence_by_frame.get(result.frame_id, [])
        weighted_auxiliary = [
            float(data["normalized_score"]) * float(fusion_weights.get(name, 0.0))
            for name, data in scores.items()
            if name != "kis" and fusion_weights.get(name, 0.0) > 0
        ]
        if not weighted_auxiliary or not evidence:
            new_score = result.score
        else:
            evidence_quality = max(
                float(item.get("document_quality", 0.0))
                * float(item.get("temporal_weight", 1.0))
                * (1.0 if item.get("exact_phrase") else 0.8)
                for item in evidence
            )
            agreement = min(1.0, len(weighted_auxiliary) / 3.0)
            new_score = (
                result.score
                + 0.20 * max(weighted_auxiliary) * evidence_quality
                + 0.02 * agreement * sum(weighted_auxiliary)
            )
        reranked.append((original_rank, replace(result, score=new_score)))
    reranked.sort(key=lambda item: (-item[1].score, item[0]))
    merged = [item[1] for item in reranked] + results[limit:]
    return _normalize_results(merged)


def _diversify_by_shot(
    results: list[SearchResult], top_k: int, limit: int = 2
) -> list[SearchResult]:
    if limit <= 0:
        return results[:top_k]
    selected: list[SearchResult] = []
    deferred: list[SearchResult] = []
    counts: dict[str, int] = {}
    for result in results:
        key = result.shot_id or result.frame_id
        if counts.get(key, 0) < limit:
            selected.append(result)
            counts[key] = counts.get(key, 0) + 1
        else:
            deferred.append(result)
        if len(selected) >= top_k:
            return selected[:top_k]
    selected.extend(deferred[: max(0, top_k - len(selected))])
    return selected[:top_k]


def _document_quality(text: str, source: str) -> float:
    compact = [character for character in text if not character.isspace()]
    if not compact:
        return 0.0
    alphanumeric = sum(character.isalnum() for character in compact)
    valid_ratio = alphanumeric / len(compact)
    minimum_length = 6 if source == "ocr" else 12
    length_factor = min(1.0, alphanumeric / minimum_length)
    return min(1.0, max(0.0, valid_ratio * length_factor))


def _score_separation(scores: list[float]) -> float:
    if not scores:
        return 0.0
    if len(scores) == 1:
        return 1.0
    ordered = sorted(scores, reverse=True)[:10]
    top = ordered[0]
    return min(1.0, max(0.0, (top - median(ordered)) / max(abs(top), 1e-9)))


def _normalize_results(results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return []
    maximum = max(result.score for result in results)
    if maximum <= 0:
        return results
    return [replace(result, score=result.score / maximum) for result in results]


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    return " ".join(
        "".join(
            character for character in decomposed if not unicodedata.combining(character)
        ).split()
    )


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = str(value).strip()
        key = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _keywords_for(processed, component: str) -> list[str]:
    originals = list(getattr(processed, f"{component}_keywords", []))
    normalized_map = getattr(processed, "normalized_keywords", {}) or {}
    return _unique((*originals, *normalized_map.get(component, [])))


def _analysis_payload(processed) -> dict:
    normalized = getattr(processed, "normalized_keywords", None)
    if normalized is None:
        normalized = {
            component: [
                _normalize_text(value) for value in getattr(processed, f"{component}_keywords", [])
            ]
            for component in ("ocr", "asr")
        }
    return {
        "visual_prompt": processed.visual_prompt,
        "ocr_keywords": processed.ocr_keywords,
        "asr_keywords": processed.asr_keywords,
        "normalized_keywords": normalized,
        "metadata": processed.metadata,
        "weights": processed.weights,
        "routing_mode": getattr(processed, "routing_mode", "fallback"),
        "modality_confidence": _modality_confidences(processed),
        "llm_status": getattr(processed, "llm_status", "unknown"),
        "fallback_reason": getattr(processed, "fallback_reason", None),
        "llm_calls": getattr(processed, "llm_calls", 0),
        "llm_attempts": getattr(processed, "llm_attempts", 0),
        "llm_usage": getattr(processed, "llm_usage", {}),
    }


def _result_payload(
    result: SearchResult,
    component_scores: dict | None = None,
    evidence: list[dict] | None = None,
) -> dict:
    return {
        "frame_id": result.frame_id,
        "video_id": result.video_id,
        "video_name": result.video_name or result.video_id,
        "frame_path": result.frame_path,
        "frame_index": result.frame_index,
        "shot_id": result.shot_id,
        "timestamp_sec": result.timestamp_sec,
        "score": round(result.score, 4),
        "component_scores": component_scores or {},
        "evidence": evidence or [],
    }


def _empty_response() -> dict:
    return {
        "results": [],
        "total": 0,
        "analysis": None,
        "component_counts": {name: 0 for name in _COMPONENTS},
        "component_status": {name: "skipped" for name in _COMPONENTS},
        "component_stats": {},
        "component_scores": {},
        "fusion_weights": {},
        "fusion_mode": "adaptive",
        "timing_ms": {"total": 0.0},
    }


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)
