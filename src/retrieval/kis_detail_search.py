"""KIS Detail 2-Stage search — Qdrant candidate retrieval + multi-concept sum fusion.

Pipeline (kis_detail_2stage_search):
  Stage 1 (general): each subquery searched via Qdrant → union candidates by
    frame_id → weighted normalized sum fusion → top_k_stage1.
  Stage 2 (specific): fetch each candidate's real vector by id, re-score
    against specific subqueries → weighted normalized sum fusion → top_k_stage2.
"""

from __future__ import annotations

import logging

import numpy as np

from config.settings import Experiment
from retrieval.hydrator import ResultHydrator
from retrieval.vqa import _get_retriever
from stores.vector.base import frame_result

LOGGER = logging.getLogger(__name__)


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

    # Reuse the cached retriever's embedder rather than building a fresh one —
    # this function runs twice per request (general + specific stages), and a
    # fresh full model load each time exhausts a small GPU within a few calls.
    retriever = _get_retriever(experiment)
    model_name = experiment.config.embedding_models[0]
    embedder = retriever.embedders[model_name]

    # ── Stage 1: general weighted normalized sum fusion via Qdrant ──
    # Each subquery is a separate Qdrant ANN search (fast, no full-corpus
    # load); candidates are unioned by frame_id, missing entries score 0
    # for that column (same as a brute-force row that never appears).
    gen_embs = np.stack(
        [np.asarray(embedder.embed_text(q), dtype="float32").flatten() for q in clean_gen]
    )
    for i in range(gen_embs.shape[0]):
        norm = np.linalg.norm(gen_embs[i])
        if norm > 1e-12:
            gen_embs[i] /= norm

    per_query_hits = [
        retriever.index.search(gen_embs[i].tolist(), top_k=top_k_stage1, model_name=model_name)
        for i in range(gen_embs.shape[0])
    ]
    frame_ids = list({hit.frame_id for hits in per_query_hits for hit in hits})
    if not frame_ids:
        return {"results": [], "total": 0}
    frame_id_to_row = {fid: row for row, fid in enumerate(frame_ids)}

    scores1 = np.zeros((len(frame_ids), len(clean_gen)), dtype="float32")
    for col, hits in enumerate(per_query_hits):
        for hit in hits:
            scores1[frame_id_to_row[hit.frame_id], col] = hit.score

    gen_weights_arr = np.array(general_weights, dtype="float32") if general_weights else None
    final1 = _weighted_sum_fusion(scores1, weights=gen_weights_arr)

    n1 = min(top_k_stage1, len(final1))
    if n1 == 0:
        return {"results": [], "total": 0}
    top_idx1 = np.argpartition(final1, -n1)[-n1:]
    top_idx1 = top_idx1[np.argsort(final1[top_idx1])[::-1]]
    top_frame_ids1 = [frame_ids[i] for i in top_idx1]

    # Stage 2 re-scores the same subset against different (specific)
    # queries, so it needs each candidate's real vector — fetched by id,
    # not the full corpus.
    cached_embs = np.stack(
        [
            np.asarray(retriever.index.get_vector(fid, model_name), dtype="float32")
            for fid in top_frame_ids1
        ]
    )

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

    final_frame_ids = [top_frame_ids1[i] for i in top_idx2]

    # ── Hydrate ─────────────────────────────────────────────────────
    hydrator = ResultHydrator(experiment)
    results = []
    for rank, frame_id in enumerate(final_frame_ids, start=1):
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
