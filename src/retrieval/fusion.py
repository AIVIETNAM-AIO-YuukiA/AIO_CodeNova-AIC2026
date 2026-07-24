"""SRRF — Score-Reflected Reciprocal Rank Fusion.

Standard Reciprocal Rank Fusion (RRF) combines multiple ranked lists using
only rank position, discarding the underlying similarity scores:

    RRF(d) = sum_i  1 / (k + rank_i(d))

This throws away information: two documents at rank 2 with scores 0.9 and 0.51
are scored identically. SRRF keeps that information by replacing the hard
integer rank with a *soft rank* — for each list, count how much every other
candidate "beats" a document, using a smooth sigmoid instead of a hard 0/1
comparison:

    soft_rank_i(d) = sum_{d' != d}  sigmoid_beta(score_i(d') - score_i(d))
    SRRF(d)        = sum_i  1 / (k + soft_rank_i(d))

sigmoid_beta(x) = 1 / (1 + exp(-beta * x)). A large beta sharpens the sigmoid
toward a hard rank comparison (recovering plain RRF); a smaller beta smooths
it, letting close scores contribute more similar soft ranks than a tied-break
hard rank would. This matches the "keep the original similarity score instead
of just rank" idea from the Cascaded System paper's visual search stage
(SigLIP + BEiT-3 -> SRRF -> BLIP-2 rerank).
"""

from __future__ import annotations

import math

from core.types import SearchResult


def _sigmoid(x: float, beta: float) -> float:
    # Guard against overflow for large |beta * x|.
    z = beta * x
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ew = math.exp(z)
    return ew / (1.0 + ew)


def _soft_ranks(scores: list[float], beta: float) -> list[float]:
    """Return the soft rank of each score within its own list (0-indexed)."""
    return [
        sum(_sigmoid(other - score, beta) for j, other in enumerate(scores) if j != i)
        for i, score in enumerate(scores)
    ]


def srrf_fuse(
    result_lists: list[list[SearchResult]],
    top_k: int,
    k: float = 60.0,
    beta: float = 10.0,
) -> list[SearchResult]:
    """Fuse multiple per-model ranked result lists into one, via SRRF.

    Each list in ``result_lists`` is one model's top-N results for the same
    query, already sorted by score descending (as returned by
    ``VectorIndex.search``). Results are matched across lists by ``frame_id``.
    Frames missing from a list are treated as having no contribution from that
    list (they do not get a soft rank penalty applied for it).

    Returns up to ``top_k`` results sorted by fused SRRF score descending. The
    returned ``SearchResult.score`` is the SRRF score (not a raw similarity),
    since the whole point is that it is not directly comparable to either
    input list's scores.
    """
    if len(result_lists) == 1:
        return result_lists[0][:top_k]

    # frame_id -> {list_index: raw_score}
    per_frame_scores: dict[str, dict[int, float]] = {}
    frame_lookup: dict[str, SearchResult] = {}
    for list_index, results in enumerate(result_lists):
        for result in results:
            per_frame_scores.setdefault(result.frame_id, {})[list_index] = result.score
            frame_lookup.setdefault(result.frame_id, result)

    fused_scores: dict[str, float] = {frame_id: 0.0 for frame_id in per_frame_scores}
    for list_index, results in enumerate(result_lists):
        if not results:
            continue
        frame_ids = [r.frame_id for r in results]
        scores = [r.score for r in results]
        for frame_id, soft_rank in zip(frame_ids, _soft_ranks(scores, beta), strict=True):
            fused_scores[frame_id] += 1.0 / (k + soft_rank)

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
