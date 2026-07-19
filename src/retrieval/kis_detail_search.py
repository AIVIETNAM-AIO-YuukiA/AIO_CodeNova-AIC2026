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
from modules.embedding import build_embedder
from retrieval.temporal_search import load_temporal_data
from retrieval.hydrator import ResultHydrator
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
        frame_embeddings, frame_records = load_temporal_data(experiment.run_dir)
    except FileNotFoundError as exc:
        return {"error": str(exc), "results": [], "total": 0}

    if frame_embeddings.shape[0] == 0 or not frame_records:
        return {"results": [], "total": 0}

    # 2. Embed từng subquery
    embedder = build_embedder(
        model_name=experiment.config.embedding_model,
        device=experiment.config.device,
    )
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

        sr = hydrator.hydrate([frame_result(frame_id, float(final_scores[idx]))])[0]

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
