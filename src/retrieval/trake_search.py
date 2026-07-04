"""TRAKE search — Bidirectional Pair Join (BPJ) algorithm.

Each adjacent pair (eᵢ, eᵢ₊₁) is processed independently:
  - Forward: from candidate eᵢ → find eᵢ₊₁ inside +5min (in-memory cosine)
  - Backward: from candidate eᵢ₊₁ → find eᵢ inside -5min (in-memory cosine)
  - Merge → top-300 pairs

Chains are formed by joining pairs on common frame_id.

Pair score:    sim(fᵢ, eᵢ) + sim(fᵢ₊₁, eᵢ₊₁)
Chain score:   mean(sim_i, sim_j, ...)
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections import Counter

import numpy as np

from config.settings import Experiment
from retrieval import build_retriever
from retrieval.hydrator import ResultHydrator
from stores.vector.base import frame_result
from retrieval.temporal_search import load_temporal_data

LOGGER = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────


def _build_video_index(
    frame_records: list[dict],
) -> dict[str, tuple[list[float], list[int]]]:
    """Build in-memory index: video_id → (sorted_timestamps, embedding_indices).

    Enables binary search over frames by timestamp within a single video.
    """
    raw: dict[str, list[tuple[float, int]]] = {}
    for i, rec in enumerate(frame_records):
        vid = rec.get("video_id")
        ts = rec.get("timestamp_sec")
        if vid is not None and ts is not None:
            raw.setdefault(vid, []).append((ts, i))
    index: dict[str, tuple[list[float], list[int]]] = {}
    for vid, pairs in raw.items():
        pairs.sort(key=lambda x: x[0])
        index[vid] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return index


def _search_event(
    retriever,
    event_text: str,
    event_index: int,
    top_k: int,
) -> list[dict]:
    """Search one event globally via Qdrant, return enriched frame dicts (distinct frame_id)."""
    results = retriever.search(query=event_text, top_k=top_k)
    out: list[dict] = []
    seen_fids: set[str] = set()
    for rank, r in enumerate(results, start=1):
        if r.frame_id in seen_fids:
            continue
        seen_fids.add(r.frame_id)
        out.append(
            {
                "event_index": event_index,
                "rank": rank,
                "score": r.score,
                "frame_id": r.frame_id,
                "video_id": r.video_id,
                "video_name": r.video_name or r.video_id,
                "frame_path": r.frame_path,
                "timestamp_sec": r.timestamp_sec,
                "shot_id": r.shot_id,
                "frame_index": r.frame_index,
            }
        )
    return out


def _embed_event(retriever, text: str) -> np.ndarray:
    """Embed event text (processed), return L2-normalized vector."""
    processed = retriever.query_processor.process(text)
    raw = retriever.embedder.embed_text(processed.visual_prompt)
    vec = np.asarray(raw, dtype="float32").flatten()
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


def _window_search_fwd(
    query_embed: np.ndarray,
    vid: str,
    t_start: float,
    window: int,
    video_index: dict,
    frame_embeddings: np.ndarray,
    frame_records: list[dict],
    hydrator: ResultHydrator,
) -> list[tuple[dict, float]]:
    """In-memory cosine search in (t_start, t_start + window].

    Returns list of (frame_info, cosine_score) sorted desc, up to 30 distinct frames.
    """
    if vid not in video_index:
        return []
    ts_list, idx_list = video_index[vid]
    lo = bisect_right(ts_list, t_start)
    hi = bisect_right(ts_list, t_start + window)
    if lo >= hi:
        return []
    idxs = idx_list[lo:hi]
    embs = frame_embeddings[idxs]
    sims = embs @ query_embed
    top_n = min(30, len(sims))
    if top_n == 0:
        return []
    top_pos = np.argpartition(sims, -top_n)[-top_n:]
    top_pos = top_pos[np.argsort(sims[top_pos])[::-1]]
    results: list[tuple[dict, float]] = []
    seen_fids: set[str] = set()
    for pos in top_pos:
        score = float(sims[pos])
        rec = frame_records[idxs[pos]]
        frame_id = rec.get("frame_id")
        if frame_id:
            if frame_id in seen_fids:
                continue
            seen_fids.add(frame_id)
            sr = hydrator.hydrate([frame_result(frame_id, score)])[0]
            info = {
                "frame_id": sr.frame_id,
                "video_id": sr.video_id,
                "video_name": sr.video_name or sr.video_id,
                "frame_path": sr.frame_path,
                "timestamp_sec": sr.timestamp_sec,
                "shot_id": sr.shot_id,
                "frame_index": sr.frame_index,
                "score": sr.score,
                "rank": len(results) + 1,
            }
        else:
            info = {
                "frame_id": None,
                "video_id": rec.get("video_id"),
                "video_name": rec.get("video_id", ""),
                "frame_path": None,
                "timestamp_sec": None,
                "shot_id": None,
                "frame_index": None,
                "score": score,
                "rank": len(results) + 1,
            }
        results.append((info, score))
        if len(seen_fids) >= 30:
            break
    return results


def _window_search_bwd(
    query_embed: np.ndarray,
    vid: str,
    t_end: float,
    window: int,
    video_index: dict,
    frame_embeddings: np.ndarray,
    frame_records: list[dict],
    hydrator: ResultHydrator,
) -> list[tuple[dict, float]]:
    """In-memory cosine search in [t_end - window, t_end).

    Returns list of (frame_info, cosine_score) sorted desc, up to 30 distinct frames.
    """
    if vid not in video_index:
        return []
    ts_list, idx_list = video_index[vid]
    lo = bisect_left(ts_list, t_end - window)
    hi = bisect_left(ts_list, t_end)
    if lo >= hi:
        return []
    idxs = idx_list[lo:hi]
    embs = frame_embeddings[idxs]
    sims = embs @ query_embed
    top_n = min(30, len(sims))
    if top_n == 0:
        return []
    top_pos = np.argpartition(sims, -top_n)[-top_n:]
    top_pos = top_pos[np.argsort(sims[top_pos])[::-1]]
    results: list[tuple[dict, float]] = []
    seen_fids: set[str] = set()
    for pos in top_pos:
        score = float(sims[pos])
        rec = frame_records[idxs[pos]]
        frame_id = rec.get("frame_id")
        if frame_id:
            if frame_id in seen_fids:
                continue
            seen_fids.add(frame_id)
            sr = hydrator.hydrate([frame_result(frame_id, score)])[0]
            info = {
                "frame_id": sr.frame_id,
                "video_id": sr.video_id,
                "video_name": sr.video_name or sr.video_id,
                "frame_path": sr.frame_path,
                "timestamp_sec": sr.timestamp_sec,
                "shot_id": sr.shot_id,
                "frame_index": sr.frame_index,
                "score": sr.score,
                "rank": len(results) + 1,
            }
        else:
            info = {
                "frame_id": None,
                "video_id": rec.get("video_id"),
                "video_name": rec.get("video_id", ""),
                "frame_path": None,
                "timestamp_sec": None,
                "shot_id": None,
                "frame_index": None,
                "score": score,
                "rank": len(results) + 1,
            }
        results.append((info, score))
        if len(seen_fids) >= 30:
            break
    return results


# ── bidirectional pair search (one adjacent pair) ──────────


def bidirectional_pair_search(
    retriever,
    event_text_i: str,
    event_text_j: str,
    event_index_i: int,
    video_index: dict,
    frame_embeddings: np.ndarray,
    frame_records: list[dict],
    hydrator: ResultHydrator,
    top_k: int = 300,
    window: int = 300,
) -> list[tuple[dict, dict, float, float, float]]:
    """Bidirectional search for one adjacent pair (eᵢ, eᵢ₊₁).

    Returns top-K pairs, each pair = (f_i_info, f_j_info, pair_score, sim_i, sim_j).
    """
    embed_i = _embed_event(retriever, event_text_i)
    embed_j = _embed_event(retriever, event_text_j)

    # 1. Forward: candidate eᵢ → find eᵢ₊₁ in +window (top-30 distinct)
    candidates_i = _search_event(retriever, event_text_i, event_index_i, top_k)
    forward: list[tuple[dict, dict, float, float, float]] = []
    for f_i in candidates_i:
        ts = f_i.get("timestamp_sec")
        vid = f_i.get("video_id")
        if ts is None or vid is None:
            continue
        for f_j_info, sim_j in _window_search_fwd(
            embed_j,
            vid,
            ts,
            window,
            video_index,
            frame_embeddings,
            frame_records,
            hydrator,
        ):
            sim_i = f_i["score"]
            pair_score = sim_i + sim_j
            forward.append((f_i, f_j_info, pair_score, sim_i, sim_j))

    # 2. Backward: candidate eᵢ₊₁ → find eᵢ in -window (top-30 distinct)
    candidates_j = _search_event(retriever, event_text_j, event_index_i + 1, top_k)
    backward: list[tuple[dict, dict, float, float, float]] = []
    for f_j in candidates_j:
        ts = f_j.get("timestamp_sec")
        vid = f_j.get("video_id")
        if ts is None or vid is None:
            continue
        for f_i_info, sim_i in _window_search_bwd(
            embed_i,
            vid,
            ts,
            window,
            video_index,
            frame_embeddings,
            frame_records,
            hydrator,
        ):
            sim_j = f_j["score"]
            pair_score = sim_i + sim_j
            backward.append((f_i_info, f_j, pair_score, sim_i, sim_j))

    # 3. Merge forward + backward, dedup by (f_i_id, f_j_id), keep higher score, filter, rank
    best: dict[tuple[str, str], tuple[dict, dict, float, float, float]] = {}
    for item in forward + backward:
        f_i, f_j, ps, si, sj = item
        f_i_fid = f_i.get("frame_id")
        f_j_fid = f_j.get("frame_id")
        if f_i_fid is None or f_j_fid is None:
            continue
        key = (f_i_fid, f_j_fid)
        if key in best and best[key][2] >= ps:
            continue
        best[key] = item

    merged = [v for v in best.values() if v[0]["timestamp_sec"] < v[1]["timestamp_sec"]]
    merged.sort(key=lambda x: -x[2])
    LOGGER.debug(
        "PairSearch (e%d,e%d): forward=%d backward=%d unique_keys=%d merged=%d returned=%d",
        event_index_i,
        event_index_i + 1,
        len(forward),
        len(backward),
        len(best),
        len(merged),
        min(len(merged), top_k),
    )
    return merged[:top_k]


# ── chain join ────────────────────────────────────────────────────────


def chain_join(
    pair_lists: list[list[tuple[dict, dict, float, float, float]]],
    window: int = 300,
) -> list[tuple[list[dict], float]]:
    """Join N-1 independent pair lists into chains of N events.

    Args:
        pair_lists: list of pair lists, pair_lists[i] = top-K pairs for (eᵢ, eᵢ₊₁)
        window: temporal window (seconds) for timestamp filtering

    Returns:
        [(chain_frames, chain_score), ...] sorted descending by score
    """
    if not pair_lists:
        return []

    # Initialize chains from the first pair list
    chains: list[tuple[list[dict], list[float]]] = [
        ([f_i, f_j], [si, sj]) for f_i, f_j, _, si, sj in pair_lists[0]
    ]

    # Set-join with subsequent pair lists: each chain picks the best match
    for pairs_k in pair_lists[1:]:
        new_chains: list[tuple[list[dict], list[float]]] = []
        for chain_frames, chain_scores in chains:
            last_id = chain_frames[-1]["frame_id"]
            last_ts = chain_frames[-1]["timestamp_sec"]
            best_ext: tuple[dict, float] | None = None  # (f_j, sj)
            best_ps = -float("inf")
            for f_i, f_j, ps, si, sj in pairs_k:
                if f_i["frame_id"] != last_id:
                    continue
                if last_ts >= f_j["timestamp_sec"]:
                    continue
                if ps > best_ps:
                    best_ps = ps
                    best_ext = (f_j, sj)
            if best_ext is not None:
                f_j, sj = best_ext
                new_chains.append((chain_frames + [f_j], chain_scores + [sj]))
        chains = new_chains
        if not chains:
            break

    # Score each chain
    scored: list[tuple[list[dict], float]] = []
    seen_chains: set[tuple[str, ...]] = set()
    for chain_frames, chain_scores in chains:
        key = tuple(f.get("frame_id") for f in chain_frames)
        if key in seen_chains:
            continue
        seen_chains.add(key)
        chain_score = sum(chain_scores) / len(chain_scores)
        scored.append((chain_frames, chain_score))

    scored.sort(key=lambda x: -x[1])
    LOGGER.debug(
        "ChainJoin: %d pair_lists → %d chains after join → %d scored",
        len(pair_lists),
        len(chains),
        len(scored),
    )
    return scored


# ── main entry point ──────────────────────────────────────────────────


def trake_search(
    experiment: Experiment,
    events: list[str],
    top_k: int = 300,
    window: int = 300,
) -> dict:
    """TRAKE pipeline — Bidirectional Pair Join (BPJ).

    Args:
        experiment: Experiment instance (config + run_dir).
        events: List of event texts, at least 2.
        top_k: Number of candidate frames per event (and pairs per adjacent pair).
        window: Temporal window (seconds), default 300 (5 minutes).

    Returns:
        Dict with "videos" (list of chains) and "total_candidates".
        Compatible with server.py response format.
    """
    if len(events) < 2:
        return {"error": "At least 2 events are required.", "videos": []}

    clean = [e.strip() for e in events if e.strip()]
    if len(clean) < 2:
        return {"error": "At least 2 non-empty events are required.", "videos": []}

    # 1. Build retriever (embedder + Qdrant)
    retriever = build_retriever(experiment)

    # 2. Load all frame embeddings + metadata
    try:
        frame_embeddings, frame_records = load_temporal_data(experiment.run_dir)
    except FileNotFoundError as exc:
        return {"error": str(exc), "videos": [], "total_candidates": 0}

    # 2b. Build in-memory video timestamp index
    video_index = _build_video_index(frame_records)

    # 3. Process each adjacent pair — fully independent
    n = len(clean)
    pair_lists: list[list[tuple[dict, dict, float, float, float]]] = []
    for i in range(n - 1):
        pairs = bidirectional_pair_search(
            retriever=retriever,
            hydrator=retriever.hydrator,
            event_text_i=clean[i],
            event_text_j=clean[i + 1],
            event_index_i=i,
            video_index=video_index,
            frame_embeddings=frame_embeddings,
            frame_records=frame_records,
            top_k=top_k,
            window=window,
        )
        pair_lists.append(pairs)

    if not pair_lists or not pair_lists[0]:
        return {"videos": [], "total_candidates": 0}

    # 4. Join into chains
    chains = chain_join(pair_lists, window)

    # 5. Final chain dedup (safety) & filter invalid chains
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[list[dict], float]] = []
    for c_frames, c_score in chains:
        fids = tuple(f.get("frame_id") for f in c_frames)
        if not all(fids) or len(set(fids)) != len(fids):
            continue
        if fids in seen:
            continue
        seen.add(fids)
        unique.append((c_frames, c_score))

    # 6. Format output — output ALL valid chains sorted by score
    out_videos: list[dict] = []
    out_vid_counter: Counter = Counter()
    for chain_frames, chain_score in unique[:top_k]:
        vids = {f.get("video_id") for f in chain_frames if f.get("video_id")}
        if len(vids) != 1:
            continue
        vid = next(iter(vids))
        vname = chain_frames[0].get("video_name", vid)
        events_out = []
        for idx, f_info in enumerate(chain_frames):
            events_out.append(
                {
                    "event_index": idx,
                    "rank": f_info.get("rank", 0),
                    "frame_id": f_info.get("frame_id"),
                    "frame_path": f_info.get("frame_path"),
                    "video_id": f_info.get("video_id"),
                    "video_name": f_info.get("video_name", f_info.get("video_id")),
                    "timestamp_sec": f_info.get("timestamp_sec"),
                    "score": f_info.get("score", 0.0),
                    "shot_id": f_info.get("shot_id"),
                    "frame_index": f_info.get("frame_index"),
                }
            )
        out_videos.append(
            {
                "video_id": vid,
                "video_name": vname,
                "score": round(chain_score, 4),
                "temporal_order_valid": True,
                "events": events_out,
            }
        )
        out_vid_counter[vid] += 1

    out_videos.sort(key=lambda x: -x["score"])
    LOGGER.info(
        "TRAKE: %d chains → %d unique → %d output from %d videos, top_vids: %s",
        len(chains),
        len(unique),
        len(out_videos),
        len(out_vid_counter),
        dict(out_vid_counter.most_common(10)),
    )

    # Print all output chains for inspection
    LOGGER.info("── TRAKE OUTPUT ──────────────────────────────────")
    for i, v in enumerate(out_videos):
        fids = [e["frame_id"] for e in v["events"]]
        timestamps = [e["timestamp_sec"] for e in v["events"]]
        scores = [round(e["score"], 4) for e in v["events"]]
        LOGGER.info(
            "  [%d] video=%s score=%.4f ts=%s scores=%s frames=%s",
            i,
            v["video_id"],
            v["score"],
            timestamps,
            scores,
            fids,
        )
    LOGGER.info("──────────────────────────────────────────────────")

    return {
        "videos": out_videos,
        "total_candidates": len(out_videos),
    }
