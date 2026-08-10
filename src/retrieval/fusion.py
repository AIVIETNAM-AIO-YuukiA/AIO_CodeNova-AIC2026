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

import numpy as np

from core.types import SearchResult
from stores.vector.base import frame_result


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


def wsf_fuse(
    model_data: dict[str, tuple[np.ndarray, list[dict], np.ndarray]],
    top_k: int,
    weights: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Weighted normalized sum fusion (WSF).

    For each model, computes cosine similarity between all frame embeddings and
    the query embedding. Scores are max-normalized globally per model. The final
    score is the weighted sum of these normalized scores across models.

    Args:
        model_data: Dict mapping model_name to (frame_embeddings, frame_records, query_embedding).
            - frame_embeddings: [N, D] array
            - frame_records: list of dicts (metadata)
            - query_embedding: [D] or [1, D] array
        top_k: Number of top frames to return.
        weights: Custom model weights. Defaults to 1/K for K models.

    Returns:
        List of SearchResult, sorted by fused WSF score descending.
    """
    if not model_data:
        return []

    model_names = list(model_data.keys())
    k_models = len(model_names)

    # Resolve weights
    if not weights:
        w = np.full(k_models, 1.0 / k_models, dtype="float32")
    else:
        w = np.array([float(weights.get(name, 1.0)) for name in model_names], dtype="float32")
        w_sum = w.sum()
        if w_sum > 1e-12:
            w = w / w_sum
        else:
            w = np.full(k_models, 1.0 / k_models, dtype="float32")

    # Assuming all models have the same number of frames N and identical frame_records
    # We take the frame_records from the first model
    first_model = model_names[0]
    _, frame_records, _ = model_data[first_model]
    n_frames = len(frame_records)

    if n_frames == 0:
        return []

    # Calculate raw cosine similarities per model
    # scores shape will be [N, K_models]
    scores = np.zeros((n_frames, k_models), dtype="float32")

    for i, model_name in enumerate(model_names):
        frame_embs, _, query_emb = model_data[model_name]
        q = np.asarray(query_emb, dtype="float32").flatten()
        # Normalize query
        norm_q = np.linalg.norm(q)
        if norm_q > 1e-12:
            q = q / norm_q

        # Calculate scores
        # frame_embs are assumed to be L2-normalized already in load_temporal_data, but let's be safe
        sim_scores = frame_embs @ q
        scores[:, i] = sim_scores

    # Max-normalize each model (column)
    max_scores = scores.max(axis=0)
    max_scores = np.where(max_scores > 1e-12, max_scores, 1.0)
    normalized = scores / max_scores

    # Weighted sum
    final_scores = normalized @ w

    # Get top-K
    n = min(top_k, n_frames)
    if n == 0:
        return []

    top_indices = np.argpartition(final_scores, -n)[-n:]
    top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

    # Convert to SearchResult
    fused: list[SearchResult] = []
    for idx in top_indices:
        rec = frame_records[idx]
        frame_id = rec.get("frame_id")
        if not frame_id:
            continue

        # We use frame_result helper to create the initial SearchResult
        sr = frame_result(frame_id, float(final_scores[idx]))
        fused.append(sr)

    return fused
