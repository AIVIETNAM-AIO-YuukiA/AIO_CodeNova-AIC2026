"""KIS Detail search — in-memory multi-concept sum fusion.

Pipeline:
  1. Load frame_embeddings (N × D)
  2. Embed từng subquery → vectors
  3. scores = frame_embeddings @ [q1, q2, ...].T  → N × K
  4. final_score = scores.sum(axis=1)
  5. Top-K frame theo final_score
  6. Hydrate frame info + return
"""

from __future__ import annotations

import logging

import numpy as np

from config.settings import Experiment
from retrieval.temporal_search import load_temporal_data
from retrieval.hydrator import ResultHydrator
from retrieval.vqa import _get_retriever
from stores.vector.base import frame_result

LOGGER = logging.getLogger(__name__)


def kis_detail_search(
    experiment: Experiment,
    subqueries: list[str],
    top_k: int = 300,
) -> dict:
    """KIS Detail pipeline — sum fusion in-memory.

    Args:
        experiment: Experiment instance.
        subqueries: List of atomic detail descriptions (already static).
        top_k: Number of top frames to return (default 300).

    Returns:
        Dict with "results" (list of hydrated frames + scores) and "total".
    """
    if not subqueries:
        return {"results": [], "total": 0}

    clean = [s.strip() for s in subqueries if s.strip()]
    if not clean:
        return {"results": [], "total": 0}

    # 1. Load frame_embeddings + metadata
    try:
        frame_embeddings, frame_records = load_temporal_data(experiment)
    except FileNotFoundError as exc:
        return {"error": str(exc), "results": [], "total": 0}

    if frame_embeddings.shape[0] == 0 or not frame_records:
        return {"results": [], "total": 0}

    # 2. Embed từng subquery
    # Reuse the cached retriever's embedder rather than building a fresh one —
    # this function runs twice per request (general + specific stages), and a
    # fresh full model load each time exhausts a small GPU within a few calls.
    embedder = _get_retriever(experiment).embedders[experiment.config.embedding_models[0]]
    query_embs = np.stack(
        [np.asarray(embedder.embed_text(q), dtype="float32").flatten() for q in clean]
    )

    # Normalize L2 cho mỗi query embedding
    for i in range(query_embs.shape[0]):
        norm = np.linalg.norm(query_embs[i])
        if norm > 1e-12:
            query_embs[i] /= norm

    # 3. Compute cosine similarity
    scores = frame_embeddings @ query_embs.T  # N × K
    if scores.shape[1] == 0:
        return {"results": [], "total": 0}

    # 4. Final score = sum
    final_scores = scores.sum(axis=1)

    # 5. Top-K
    n = min(top_k, len(final_scores))
    if n == 0:
        return {"results": [], "total": 0}

    top_indices = np.argpartition(final_scores, -n)[-n:]
    top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

    # 6. Hydrate frame info
    hydrator = ResultHydrator(experiment)
    results = []
    for idx in top_indices:
        rec = frame_records[idx]
        frame_id = rec.get("frame_id")
        if not frame_id:
            continue

        hydrated = hydrator.hydrate([frame_result(frame_id, float(final_scores[idx]))])
        if not hydrated:
            continue
        sr = hydrated[0]

        sub_scores = {}
        for i, q in enumerate(clean):
            sub_scores[f"sub_{i}"] = round(float(scores[idx, i]), 4)

        results.append(
            {
                "frame_id": sr.frame_id,
                "video_id": sr.video_id,
                "video_name": sr.video_name or sr.video_id,
                "frame_path": sr.frame_path,
                "frame_index": sr.frame_index,
                "shot_id": sr.shot_id,
                "timestamp_sec": sr.timestamp_sec,
                "score": round(float(final_scores[idx]), 4),
                "sub_scores": sub_scores,
            }
        )

    LOGGER.info("KIS Detail: %d subqueries → %d frames (top %d)", len(clean), len(results), n)
    return {"results": results, "total": len(results)}


def _weighted_sum_fusion(scores: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted normalized sum fusion.

    For each subquery column:
      1. Find maxScore = max over all frames for that subquery.
      2. Normalize each frame's score by maxScore.
      3. Multiply by the subquery's weight.
      4. Sum across subqueries for each frame's final score.

    Args:
        scores: [N_frames, K_subqueries] raw cosine matrix.
        weights: [K_subqueries] or None (defaults to 1/K).

    Returns:
        [N_frames] final scores after weighted normalized fusion.
    """
    if scores.ndim == 1 or scores.shape[1] == 1:
        return scores.flatten()

    k = scores.shape[1]
    if weights is None:
        weights = np.full(k, 1.0 / k, dtype="float32")
    else:
        weights = np.asarray(weights, dtype="float32")
        w_sum = weights.sum()
        if w_sum > 1e-12:
            weights = weights / w_sum
        else:
            weights = np.full(k, 1.0 / k, dtype="float32")

    max_scores = scores.max(axis=0)
    max_scores = np.where(max_scores > 1e-12, max_scores, 1.0)

    normalized = scores / max_scores
    final = normalized @ weights

    return final


def kis_detail_2stage_search(
    experiment: Experiment,
    general: list[str],
    specific: list[str],
    top_k_stage1: int = 1000,
    top_k_stage2: int = 300,
    general_weights: list[float] | None = None,
    specific_weights: list[float] | None = None,
) -> dict:
    """KIS Detail 2-Stage — weighted normalized sum fusion.

    Stage 1 (general): weighted normalized sum fusion → top *top_k_stage1*.
    Stage 2 (specific): weighted normalized sum fusion on cached Stage-1 → top *top_k_stage2*.

    Default weights = 1/N. Future frontend will pass custom weights.

    Args:
        experiment: Experiment instance.
        general: Coarse visual subqueries (Stage 1).
        specific: Fine-grained visual subqueries (Stage 2).
        top_k_stage1: Frames to keep after Stage 1 (default 1000).
        top_k_stage2: Frames to return after Stage 2 (default 300).
        general_weights: Custom weights for general subqueries.
        specific_weights: Custom weights for specific subqueries.

    Returns:
        Dict with "results" (list of hydrated frames) and "total".
    """
    if not general or not specific:
        return {"results": [], "total": 0}

    clean_gen = [s.strip() for s in general if s.strip()]
    clean_spec = [s.strip() for s in specific if s.strip()]
    if not clean_gen or not clean_spec:
        return {"results": [], "total": 0}

    # 1. Load data
    try:
        frame_embeddings, frame_records = load_temporal_data(experiment)
    except FileNotFoundError as exc:
        return {"error": str(exc), "results": [], "total": 0}

    if frame_embeddings.shape[0] == 0 or not frame_records:
        return {"results": [], "total": 0}

    # Reuse the cached retriever's embedder rather than building a fresh one —
    # this function runs twice per request (general + specific stages), and a
    # fresh full model load each time exhausts a small GPU within a few calls.
    embedder = _get_retriever(experiment).embedders[experiment.config.embedding_models[0]]

    # ── Stage 1: general weighted normalized sum fusion ─────────────
    gen_embs = np.stack(
        [np.asarray(embedder.embed_text(q), dtype="float32").flatten() for q in clean_gen]
    )
    for i in range(gen_embs.shape[0]):
        norm = np.linalg.norm(gen_embs[i])
        if norm > 1e-12:
            gen_embs[i] /= norm

    scores1 = frame_embeddings @ gen_embs.T  # [all, K_gen]
    gen_weights_arr = np.array(general_weights, dtype="float32") if general_weights else None
    final1 = _weighted_sum_fusion(scores1, weights=gen_weights_arr)

    n1 = min(top_k_stage1, len(final1))
    if n1 == 0:
        return {"results": [], "total": 0}
    top_idx1 = np.argpartition(final1, -n1)[-n1:]
    top_idx1 = top_idx1[np.argsort(final1[top_idx1])[::-1]]

    cached_embs = frame_embeddings[top_idx1]  # [n1, D]

    LOGGER.info(
        "KIS Detail 2-Stage | Stage 1: %d general → %d candidates",
        len(clean_gen),
        n1,
    )

    # ── Stage 2: specific weighted normalized sum fusion ─────────────
    spec_embs = np.stack(
        [np.asarray(embedder.embed_text(q), dtype="float32").flatten() for q in clean_spec]
    )
    for i in range(spec_embs.shape[0]):
        norm = np.linalg.norm(spec_embs[i])
        if norm > 1e-12:
            spec_embs[i] /= norm

    scores2 = cached_embs @ spec_embs.T  # [n1, K_spec]
    spec_weights_arr = np.array(specific_weights, dtype="float32") if specific_weights else None
    final2 = _weighted_sum_fusion(scores2, weights=spec_weights_arr)

    n2 = min(top_k_stage2, len(final2))
    if n2 == 0:
        return {"results": [], "total": 0}
    top_idx2 = np.argpartition(final2, -n2)[-n2:]
    top_idx2 = top_idx2[np.argsort(final2[top_idx2])[::-1]]

    final_indices = top_idx1[top_idx2]

    # ── Hydrate ─────────────────────────────────────────────────────
    hydrator = ResultHydrator(experiment)
    results = []
    for rank, idx in enumerate(final_indices, start=1):
        rec = frame_records[idx]
        frame_id = rec.get("frame_id")
        if not frame_id:
            continue

        hydrated = hydrator.hydrate([frame_result(frame_id, float(final2[rank - 1]))])
        if not hydrated:
            continue
        sr = hydrated[0]

        sub_scores = {}
        for i, q in enumerate(clean_spec):
            sub_scores[f"sub_{i}"] = round(float(scores2[top_idx2[rank - 1], i]), 4)

        results.append(
            {
                "frame_id": sr.frame_id,
                "video_id": sr.video_id,
                "video_name": sr.video_name or sr.video_id,
                "frame_path": sr.frame_path,
                "frame_index": sr.frame_index,
                "shot_id": sr.shot_id,
                "timestamp_sec": sr.timestamp_sec,
                "score": round(float(final2[rank - 1]), 4),
                "sub_scores": sub_scores,
            }
        )

    LOGGER.info(
        "KIS Detail 2-Stage | Stage 2: %d specific → %d final frames",
        len(clean_spec),
        len(results),
    )
    return {"results": results, "total": len(results)}
