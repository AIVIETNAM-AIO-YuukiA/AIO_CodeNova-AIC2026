"""TRAKE search — Bidirectional Pair Join (BPJ) algorithm for Bản 1.

Each adjacent pair (eᵢ, eᵢ₊₁) is processed independently:
  - Forward: candidate eᵢ from KIS Basic search → find eᵢ₊₁ inside +window (in-memory cosine)
  - Backward: candidate eᵢ₊₁ from KIS Basic search → find eᵢ inside -window (in-memory cosine)
  - Merge → top pairs

Chains are formed by joining pairs on common frame_id.

Pair score:    sim(fᵢ, eᵢ) + sim(fᵢ₊₁, eᵢ₊₁)
Chain score:   mean(sim_i, sim_j, ...)
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections import Counter
import re

import numpy as np

from config.settings import Experiment
from retrieval.hydrator import ResultHydrator
from stores.vector.base import frame_result
from retrieval.temporal_search import load_temporal_data
from retrieval.intelligent_search import intelligent_search
from stores.text.factory import build_text_index
from retrieval.text_search import NearestFrameIndex

LOGGER = logging.getLogger(__name__)


def _get_timestamp_sec(rec: dict) -> float:
    """Get timestamp_sec from record metadata (provided by map-keyframes JSON)."""
    ts = rec.get("timestamp_sec")
    if ts is not None:
        return float(ts)
    return 0.0


def _build_video_index(
    frame_records: list[dict],
) -> dict[str, tuple[list[float], list[int]]]:
    """Build in-memory index: video_id → (sorted_timestamps, embedding_indices).

    Enables binary search over frames by timestamp within a single video.
    """
    raw: dict[str, list[tuple[float, int]]] = {}
    for i, rec in enumerate(frame_records):
        vid = rec.get("video_id")
        ts = _get_timestamp_sec(rec)
        rec["timestamp_sec"] = ts
        if vid is not None:
            raw.setdefault(vid, []).append((ts, i))
    index: dict[str, tuple[list[float], list[int]]] = {}
    for vid, pairs in raw.items():
        pairs.sort(key=lambda x: x[0])
        index[vid] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return index


def _fetch_text_scores_to_ram(experiment: Experiment, processed) -> dict:
    """Fetch global BM25 scores into RAM for fast window lookup."""
    index = build_text_index(experiment)
    nearest = NearestFrameIndex(experiment)

    def _fetch(keywords, source):
        if not keywords:
            return {}, 0.0
        best_score = {}
        for kw in keywords:
            for doc in index.search_documents(kw, top_k=5000, source=source):
                fid = doc.get("frame_id")
                if not fid and source == "asr":
                    fid = nearest.nearest(doc.get("video_id", ""), doc.get("timestamp_sec") or 0.0)
                if not fid:
                    continue
                s = float(doc.get("score", 0.0))
                if s > best_score.get(fid, -1.0):
                    best_score[fid] = s
        m_score = max(best_score.values()) if best_score else 0.0
        return best_score, m_score

    ocr_d, max_ocr = _fetch(processed.ocr_keywords, "ocr")
    asr_d, max_asr = _fetch(processed.asr_keywords, "asr")
    cap_d, max_cap = _fetch(processed.caption_keywords, "caption")

    weights = processed.weights
    fusion_weights = {
        "ocr": weights.get("ocr_bonus", 0.0),
        "asr": weights.get("asr_bonus", 0.0),
        "caption": weights.get("caption_bonus", 0.0),
    }

    return {
        "ocr_dict": ocr_d,
        "max_ocr": max_ocr,
        "asr_dict": asr_d,
        "max_asr": max_asr,
        "cap_dict": cap_d,
        "max_cap": max_cap,
        "fusion_weights": fusion_weights,
    }


def _search_event_intelligent(
    experiment: Experiment,
    event_text: str,
    event_index: int,
    top_k: int = 300,
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
    use_llm: bool = True,
) -> list[dict]:
    """Search one event globally using intelligent_search."""
    response = intelligent_search(
        experiment=experiment,
        query=event_text,
        top_k=top_k,
        enable_kis=True,
        enable_ocr=True,
        enable_asr=True,
        enable_caption=True,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
    )
    out = []
    seen_fids = set()
    for rank, r in enumerate(response.get("results", []), start=1):
        if r["frame_id"] in seen_fids:
            continue
        seen_fids.add(r["frame_id"])
        ts = r.get("timestamp_sec")
        if ts is None or ts == 0.0:
            ts = _get_timestamp_sec(
                {
                    "frame_id": r["frame_id"],
                    "timestamp_sec": r.get("timestamp_sec"),
                    "frame_index": r.get("frame_index"),
                }
            )
        out.append(
            {
                "event_index": event_index,
                "rank": rank,
                "score": r["score"],
                "frame_id": r["frame_id"],
                "video_id": r["video_id"],
                "video_name": r.get("video_name") or r["video_id"],
                "frame_path": r["frame_path"],
                "timestamp_sec": ts,
                "shot_id": r.get("shot_id"),
                "frame_index": r.get("frame_index"),
            }
        )
    return out


def _embed_event(
    retriever,
    text: str,
    enabled_models: list[str] | None = None,
    use_llm: bool = True,
) -> np.ndarray:
    """Embed event text using the first active model from enabled_models.

    Returns an L2-normalized vector for in-memory temporal window cosine search.
    """
    processed = retriever.query_processor.process(
        text, enabled_models=enabled_models, use_llm=use_llm
    )
    active = retriever._select_embedders(enabled_models)
    first_model = list(active.keys())[0]
    embedder = active[first_model]

    if "vietnamese" in first_model.lower() or "vism" in first_model.lower():
        visual_query = processed.visual_prompt_vi
    else:
        visual_query = processed.visual_prompt

    raw = embedder.embed_text(visual_query)
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
    text_context: dict | None = None,
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

    final_scores = np.copy(sims)
    if text_context:
        w_ocr = text_context["fusion_weights"]["ocr"]
        w_asr = text_context["fusion_weights"]["asr"]
        w_cap = text_context["fusion_weights"]["caption"]

        m_ocr = max(text_context["max_ocr"], 15.0)
        m_asr = max(text_context["max_asr"], 15.0)
        m_cap = max(text_context["max_cap"], 15.0)

        for i, pos in enumerate(idxs):
            rec = frame_records[pos]
            fid = rec.get("frame_id")
            if not fid:
                continue

            ocr_s = text_context["ocr_dict"].get(fid, 0.0) / m_ocr
            asr_s = text_context["asr_dict"].get(fid, 0.0) / m_asr
            cap_s = text_context["cap_dict"].get(fid, 0.0) / m_cap

            bonus = (w_ocr * ocr_s) + (w_asr * asr_s) + (w_cap * cap_s)
            final_scores[i] += bonus

        global_max_possible = 1.0 + w_ocr + w_asr + w_cap
        if global_max_possible > 0:
            final_scores = final_scores / global_max_possible

    top_n = min(30, len(final_scores))
    if top_n == 0:
        return []
    top_pos = np.argpartition(final_scores, -top_n)[-top_n:]
    top_pos = top_pos[np.argsort(final_scores[top_pos])[::-1]]
    results: list[tuple[dict, float]] = []
    seen_fids: set[str] = set()
    for pos in top_pos:
        score = float(final_scores[pos])
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
    text_context: dict | None = None,
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

    final_scores = np.copy(sims)
    if text_context:
        w_ocr = text_context["fusion_weights"]["ocr"]
        w_asr = text_context["fusion_weights"]["asr"]
        w_cap = text_context["fusion_weights"]["caption"]

        m_ocr = max(text_context["max_ocr"], 15.0)
        m_asr = max(text_context["max_asr"], 15.0)
        m_cap = max(text_context["max_cap"], 15.0)

        for i, pos in enumerate(idxs):
            rec = frame_records[pos]
            fid = rec.get("frame_id")
            if not fid:
                continue

            ocr_s = text_context["ocr_dict"].get(fid, 0.0) / m_ocr
            asr_s = text_context["asr_dict"].get(fid, 0.0) / m_asr
            cap_s = text_context["cap_dict"].get(fid, 0.0) / m_cap

            bonus = (w_ocr * ocr_s) + (w_asr * asr_s) + (w_cap * cap_s)
            final_scores[i] += bonus

        global_max_possible = 1.0 + w_ocr + w_asr + w_cap
        if global_max_possible > 0:
            final_scores = final_scores / global_max_possible

    top_n = min(30, len(final_scores))
    if top_n == 0:
        return []
    top_pos = np.argpartition(final_scores, -top_n)[-top_n:]
    top_pos = top_pos[np.argsort(final_scores[top_pos])[::-1]]
    results: list[tuple[dict, float]] = []
    seen_fids: set[str] = set()
    for pos in top_pos:
        score = float(final_scores[pos])
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
    experiment: Experiment,
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
    *,
    sub_details_i: list[str] | None = None,
    sub_details_j: list[str] | None = None,
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
    use_llm: bool = True,
) -> list[tuple[dict, dict, float, float, float]]:
    """Bidirectional search for one adjacent pair (eᵢ, eᵢ₊₁).

    Returns top-K pairs, each pair = (f_i_info, f_j_info, pair_score, sim_i, sim_j).
    """
    processed_i = retriever.query_processor.process(
        event_text_i, enabled_models=enabled_models, use_llm=use_llm
    )
    processed_j = retriever.query_processor.process(
        event_text_j, enabled_models=enabled_models, use_llm=use_llm
    )

    text_context_i = _fetch_text_scores_to_ram(experiment, processed_i)
    text_context_j = _fetch_text_scores_to_ram(experiment, processed_j)

    embed_i = _embed_event(retriever, event_text_i, enabled_models=enabled_models, use_llm=use_llm)
    embed_j = _embed_event(retriever, event_text_j, enabled_models=enabled_models, use_llm=use_llm)

    # 1. Candidate search for eᵢ and eᵢ₊₁ via full Intelligent pipeline (top 300 pool)
    candidate_pool = top_k
    candidates_i = _search_event_intelligent(
        experiment,
        event_text_i,
        event_index_i,
        candidate_pool,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
    )
    candidates_j = _search_event_intelligent(
        experiment,
        event_text_j,
        event_index_i + 1,
        candidate_pool,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
    )

    rank_map_i = {f["frame_id"]: f["rank"] for f in candidates_i if f.get("frame_id")}
    rank_map_j = {f["frame_id"]: f["rank"] for f in candidates_j if f.get("frame_id")}

    # 2. Direct KIS-to-KIS Pairs (both frames come directly from full KIS Basic pipeline)
    vids_j: dict[str, list[dict]] = {}
    for f_j in candidates_j:
        vid = f_j.get("video_id")
        if vid:
            vids_j.setdefault(vid, []).append(f_j)

    direct_pairs: list[tuple[dict, dict, float, float, float]] = []
    for f_i in candidates_i:
        vid = f_i.get("video_id")
        ts_i = f_i.get("timestamp_sec")
        if not vid or ts_i is None or vid not in vids_j:
            continue
        for f_j in vids_j[vid]:
            ts_j = f_j.get("timestamp_sec")
            if ts_j is not None and ts_j > ts_i and (ts_j - ts_i) <= window:
                sim_i = f_i["score"]
                sim_j = f_j["score"]
                pair_score = sim_i + sim_j
                direct_pairs.append((f_i, f_j, pair_score, sim_i, sim_j))

    # 3. Forward: candidate eᵢ → find eᵢ₊₁ in +window (top-30 distinct)
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
            text_context_j,
        ):
            fid_j = f_j_info.get("frame_id")
            if fid_j and fid_j in rank_map_j:
                f_j_info["rank"] = rank_map_j[fid_j]
            else:
                f_j_info["rank"] = f">{candidate_pool}"
            sim_i = f_i["score"]
            pair_score = sim_i + sim_j
            forward.append((f_i, f_j_info, pair_score, sim_i, sim_j))

    # 4. Backward: candidate eᵢ₊₁ → find eᵢ in -window (top-30 distinct)
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
            text_context_i,
        ):
            fid_i = f_i_info.get("frame_id")
            if fid_i and fid_i in rank_map_i:
                f_i_info["rank"] = rank_map_i[fid_i]
            else:
                f_i_info["rank"] = f">{candidate_pool}"
            sim_j = f_j["score"]
            pair_score = sim_i + sim_j
            backward.append((f_i_info, f_j, pair_score, sim_i, sim_j))

    # 5. Merge direct + forward + backward, dedup by (f_i_id, f_j_id), keep higher score, filter, rank
    best: dict[tuple[str, str], tuple[dict, dict, float, float, float]] = {}
    for item in direct_pairs + forward + backward:
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


def _get_shot_num(sid: str | None) -> int | None:
    if not sid:
        return None
    nums = re.findall(r"\d+", sid)
    return int(nums[-1]) if nums else None


def chain_join(
    pair_lists: list[list[tuple[dict, dict, float, float, float]]],
    window: int = 15,
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
            last_frame = chain_frames[-1]
            last_vid = last_frame.get("video_id")
            last_shot = last_frame.get("shot_id")
            last_shot_num = _get_shot_num(last_shot)
            last_ts = last_frame.get("timestamp_sec")

            best_ext: tuple[dict, float] | None = None  # (f_j, sj)
            best_ps = -float("inf")
            for f_i, f_j, ps, si, sj in pairs_k:
                # 0. Ensure the joining point is within 3 shots
                f_i_shot = f_i.get("shot_id")
                f_i_shot_num = _get_shot_num(f_i_shot)

                if last_shot_num is not None and f_i_shot_num is not None:
                    if abs(f_i_shot_num - last_shot_num) > 3:
                        continue
                elif last_shot and f_i_shot and f_i_shot != last_shot:
                    continue

                # 1. Next event frame f_j must be in the same video
                if f_j.get("video_id") != last_vid:
                    continue
                # 2. Next event f_j must occur AFTER last_ts (previous event's timestamp)
                ts_j = f_j.get("timestamp_sec")
                if last_ts is not None and ts_j is not None:
                    if ts_j <= last_ts:
                        continue
                    # 3. Next event f_j must occur within window seconds after last_ts
                    if ts_j - last_ts > window:
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

    # Max-Normalized Sum Fusion (Bản 2 style)
    num_events = len(chains[0][1]) if chains else 0
    max_scores = [0.0] * num_events
    for _, event_scores in chains:
        for k in range(num_events):
            if event_scores[k] > max_scores[k]:
                max_scores[k] = event_scores[k]

    for k in range(num_events):
        if max_scores[k] <= 1e-12:
            max_scores[k] = 1.0

    scored: list[tuple[list[dict], float]] = []
    seen_chains: set[tuple[str, ...]] = set()
    for chain_frames, chain_scores in chains:
        key = tuple(f.get("frame_id") for f in chain_frames)
        if key in seen_chains:
            continue
        seen_chains.add(key)

        # Normalize each event score by its max_score across candidates
        norm_scores = [chain_scores[k] / max_scores[k] for k in range(num_events)]
        chain_score = sum(norm_scores) / num_events
        scored.append((chain_frames, float(chain_score)))

    scored.sort(key=lambda x: -x[1])
    LOGGER.debug(
        "ChainJoin: %d pair_lists → %d chains after join → %d scored",
        len(pair_lists),
        len(chains),
        len(scored),
    )
    return scored


# ── main entry point ──────────────────────────────────────────────────


def trake_bpj_search(
    experiment: Experiment,
    events: list[str] | list[dict],
    top_k: int = 300,
    window: int = 15,
    enabled_models: list[str] | None = None,
    use_reranker: bool | None = None,
    use_llm: bool = True,
) -> dict:
    """TRAKE pipeline — Bidirectional Pair Join (BPJ) for Bản 1.

    Args:
        experiment: Experiment instance (config + run_dir).
        events: List of event texts (str) or event objects ``{"text": ..., "sub_details": [...]}``, at least 2.
        top_k: Number of candidate frames per event (and pairs per adjacent pair).
        window: Temporal window (seconds), default 15.
        enabled_models: List of selected model names from UI checkboxes.
        use_reranker: Whether cross-encoder reranking is enabled.

    Returns:
        Dict with "videos" (list of chains) and "total_candidates".
        Compatible with server.py response format.
    """
    # Normalize events: always list of dicts
    parsed_events: list[dict] = []
    sub_details_list: list[list[str]] = []
    for e in events:
        if isinstance(e, str):
            text = e.strip()
            parsed_events.append({"text": text, "sub_details": []} if text else None)
            sub_details_list.append([])
        elif isinstance(e, dict):
            text = e.get("text", "").strip()
            parsed_events.append(
                {"text": text, "sub_details": e.get("sub_details", [])} if text else None
            )
            sub_details_list.append(e.get("sub_details", []))
        else:
            parsed_events.append(None)
            sub_details_list.append([])
    parsed_events = [p for p in parsed_events if p is not None]
    sub_details_list = [s for s, p in zip(sub_details_list, events) if p is not None]

    if len(parsed_events) < 2:
        return {"error": "At least 2 non-empty events are required.", "videos": []}

    clean = [p["text"] for p in parsed_events]

    LOGGER.info(
        "TRAKE BPJ search: %d events, window=%ds, top_k=%d, models=%s, reranker=%s",
        len(clean),
        window,
        top_k,
        enabled_models,
        use_reranker,
    )

    # 1. Reuse cached retriever
    from retrieval.vqa import _get_retriever

    retriever = _get_retriever(experiment)

    # 2. Load all frame embeddings + metadata (first active model)
    try:
        active_models = retriever._select_embedders(enabled_models)
        first_model = list(active_models.keys())[0]
        frame_embeddings, frame_records = load_temporal_data(experiment, model_name=first_model)
        for rec in frame_records:
            rec["timestamp_sec"] = _get_timestamp_sec(rec)
    except FileNotFoundError as exc:
        return {"error": str(exc), "videos": [], "total_candidates": 0}

    # 2b. Build in-memory video timestamp index
    video_index = _build_video_index(frame_records)
    hydrator = ResultHydrator(experiment)

    # 3. Process each adjacent pair — fully independent
    n = len(parsed_events)
    pair_lists: list[list[tuple[dict, dict, float, float, float]]] = []
    for i in range(n - 1):
        pairs = bidirectional_pair_search(
            experiment=experiment,
            retriever=retriever,
            event_text_i=clean[i],
            event_text_j=clean[i + 1],
            event_index_i=i,
            video_index=video_index,
            frame_embeddings=frame_embeddings,
            frame_records=frame_records,
            top_k=top_k,
            window=window,
            sub_details_i=sub_details_list[i],
            sub_details_j=sub_details_list[i + 1],
            hydrator=hydrator,
            enabled_models=enabled_models,
            use_reranker=use_reranker,
            use_llm=use_llm,
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
        "TRAKE BPJ: %d chains → %d unique → %d output from %d videos, top_vids: %s",
        len(chains),
        len(unique),
        len(out_videos),
        len(out_vid_counter),
        dict(out_vid_counter.most_common(10)),
    )

    return {
        "videos": out_videos,
        "total_candidates": len(out_videos),
    }
