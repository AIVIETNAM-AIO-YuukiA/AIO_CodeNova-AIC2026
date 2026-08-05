"""SRRF — Score-Reflected Reciprocal Rank Fusion.

Ported 1:1 from the AIC_2025 reference project's
``online/backend/fusion.py::srrf_weighted_fuse``: per-model min-max score
normalization blended with classic RRF rank terms.

    normalized_score_i(d) = (score_i(d) - min_i) / (max_i - min_i)
    srrf_i(d)              = (1 - beta) * 1/(k + rank_i(d)) + beta * normalized_score_i(d)
    SRRF(d)                = sum_i  weight_i * srrf_i(d)

``beta`` controls the RRF-vs-score blend (0 = pure RRF, 1 = pure normalized
score); ``weights`` lets individual models contribute more or less to the
fused score.
"""

from __future__ import annotations

from core.types import SearchResult


def srrf_fuse(
    results_by_model: dict[str, list[SearchResult]],
    top_k: int,
    k: float = 60.0,
    beta: float = 0.5,
    weights: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Fuse multiple per-model ranked result lists into one, via SRRF.

    Each list in ``results_by_model`` is one model's top-N results for the
    same query, already sorted by score descending (as returned by
    ``VectorIndex.search``). Results are matched across lists by ``frame_id``.

    Returns up to ``top_k`` results sorted by fused SRRF score descending. The
    returned ``SearchResult.score`` is the SRRF score (not a raw similarity),
    since the whole point is that it is not directly comparable to any input
    list's scores.
    """
    if len(results_by_model) == 1:
        ((_, only_results),) = results_by_model.items()
        return only_results[:top_k]

    weights = weights or {}
    fused_scores: dict[str, float] = {}
    frame_lookup: dict[str, SearchResult] = {}

    for model_name, results in results_by_model.items():
        if not results:
            continue
        w = float(weights.get(model_name, 1.0))

        scores = [r.score for r in results]
        min_score, max_score = min(scores), max(scores)
        score_range = max(max_score - min_score, 1e-9)

        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        for rank, result in enumerate(sorted_results, start=1):
            normalized_score = (result.score - min_score) / score_range
            rank_term = 1.0 / (k + rank)
            srrf_term = (1 - beta) * rank_term + beta * normalized_score

            fused_scores[result.frame_id] = fused_scores.get(result.frame_id, 0.0) + w * srrf_term

            existing = frame_lookup.get(result.frame_id)
            if existing is None or result.score > existing.score:
                frame_lookup[result.frame_id] = result

    ranked_ids = sorted(fused_scores, key=lambda fid: fused_scores[fid], reverse=True)[:top_k]
    fused: list[SearchResult] = []
    for frame_id in ranked_ids:
        base = frame_lookup[frame_id]
        fused.append(
            SearchResult(
                frame_id=base.frame_id,
                video_id=base.video_id,
                score=fused_scores[frame_id],
                frame_path=base.frame_path,
                video_path=base.video_path,
                video_name=base.video_name,
                shot_id=base.shot_id,
                frame_index=base.frame_index,
                timestamp_sec=base.timestamp_sec,
            )
        )
    return fused
