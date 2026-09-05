"""Backend API handlers and data helpers for CodeNova UI."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote
import json
import re

from config.settings import Experiment
from core.logging import get_logger
from core.types import SearchResult
from retrieval.intelligent_search import intelligent_search
from retrieval.kis_detail_search import kis_detail_2stage_search
from retrieval.text_search import AsrTemporalMapper, text_search
from retrieval.tracks import SUPPORTED_TRACKS, TrackQuery, build_retrieval_text
from retrieval.vqa import enhanced_temporal_search, trake_search, vqa_search
import numpy as np

LOGGER = get_logger(__name__)


def _parse_json_bool(value: object, *, field: str, default: bool | None) -> bool | None:
    """Accept JSON booleans only; avoid Python's ``bool("false")`` trap."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a boolean")


def _attach_nested_frame_image_urls(value: object) -> None:
    """Add ``image_url`` to nested frame references in a JSON response."""
    if isinstance(value, list):
        for item in value:
            _attach_nested_frame_image_urls(item)
        return
    if not isinstance(value, dict):
        return

    frame_path = value.get("frame_path")
    if isinstance(frame_path, str) and frame_path:
        value.setdefault("image_url", f"/frame?path={quote(frame_path)}")

    frame_paths = value.get("frame_paths")
    if isinstance(frame_paths, list):
        value.setdefault(
            "image_urls",
            [f"/frame?path={quote(fp)}" for fp in frame_paths if isinstance(fp, str) and fp],
        )

    for nested in value.values():
        _attach_nested_frame_image_urls(nested)


def warmup_models(reranker, experiment: Experiment, retriever=None) -> None:
    """Pre-load heavy models before the server starts accepting requests.

    The lazy-loaded models (BLIP-2 reranker, the embedders) can take minutes to
    download and place on the GPU. Loading them up front keeps the first query
    from paying that cost, and — because this runs to completion before the
    listener starts — no two threads can race to initialize the same model.
    """
    LOGGER.info("[warmup] Pre-loading models...")
    try:
        if reranker is not None and hasattr(reranker, "_load"):
            LOGGER.info("[warmup] Loading BLIP-2 reranker...")
            reranker._load()
            LOGGER.info("[warmup] BLIP-2 reranker ready.")
    except Exception as exc:
        LOGGER.warning("[warmup] Reranker pre-load failed (non-fatal): %s", exc)

    try:
        # Warm every embedder the retriever will actually use, via the same
        # instances, so the first query finds them already resident.
        if retriever is not None:
            for model_name, embedder in retriever.embedders.items():
                embedder.embed_text("warmup query")
                LOGGER.info("[warmup] Embedder ready: %s", model_name)
    except Exception as exc:
        LOGGER.warning("[warmup] Embedder pre-load failed (non-fatal): %s", exc)

    LOGGER.info("[warmup] All models pre-loaded and ready.")


def ensure_manifests(experiment: Experiment) -> None:
    """Auto-recover manifests/frames.jsonl and videos.jsonl if missing or empty."""
    manifests_dir = experiment.run_dir / "manifests"
    frames_jsonl = manifests_dir / "frames.jsonl"
    videos_jsonl = manifests_dir / "videos.jsonl"

    if frames_jsonl.exists() and frames_jsonl.stat().st_size > 0:
        return

    LOGGER.info("[manifests] Missing or empty frames.jsonl detected. Generating automatically...")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = experiment.run_dir / "embeddings"
    json_ids_files = list(embeddings_dir.glob("frame_ids*.json")) if embeddings_dir.exists() else []

    frames_records = []
    videos_records = set()

    if json_ids_files:
        json_ids_file = json_ids_files[0]
        LOGGER.info("[manifests] Reading frame IDs from %s...", json_ids_file.name)
        try:
            with open(json_ids_file, encoding="utf-8") as f:
                frame_ids = json.load(f)

            for fid in frame_ids:
                parts = fid.split("_")
                video_id = parts[0]
                videos_records.add(video_id)
                img_name = fid + ".jpg" if not fid.endswith(".jpg") else fid
                expected_path = experiment.run_dir / "frames" / video_id / img_name
                rel_path = (
                    expected_path.relative_to(Path.cwd())
                    if expected_path.is_relative_to(Path.cwd())
                    else expected_path
                )

                match = re.search(r"_f(\d+)", fid)
                f_num = int(match.group(1)) if match else 0
                ts = None

                frames_records.append(
                    {
                        "frame_id": fid,
                        "video_id": video_id,
                        "shot_id": parts[1] if len(parts) > 2 else f"{video_id}_s0",
                        "frame_index": f_num,
                        "timestamp_sec": ts,
                        "frame_path": str(rel_path).replace("\\", "/"),
                    }
                )
        except Exception as exc:
            LOGGER.warning("[manifests] Failed to parse %s: %s", json_ids_file, exc)

    if not frames_records:
        frames_dir = experiment.run_dir / "frames"
        if frames_dir.exists():
            LOGGER.info("[manifests] Scanning frames directory %s...", frames_dir)
            for img_path in frames_dir.glob("*/*.jpg"):
                video_id = img_path.parent.name
                videos_records.add(video_id)
                frame_id = f"{video_id}_{img_path.stem}"
                match = re.search(r"_f(\d+)", frame_id)
                f_num = int(match.group(1)) if match else 0
                ts = None

                frames_records.append(
                    {
                        "frame_id": frame_id,
                        "video_id": video_id,
                        "shot_id": f"{video_id}_s0",
                        "frame_index": f_num,
                        "timestamp_sec": ts,
                        "frame_path": str(img_path).replace("\\", "/"),
                    }
                )

    if frames_records:
        with open(frames_jsonl, "w", encoding="utf-8") as f:
            for r in frames_records:
                f.write(json.dumps(r) + "\n")

        with open(videos_jsonl, "w", encoding="utf-8") as f:
            for v_id in videos_records:
                f.write(
                    json.dumps(
                        {
                            "video_id": v_id,
                            "path": f"data/videos/{v_id}.mp4",
                            "checksum": "dummy_checksum",
                            "size_bytes": 0,
                        }
                    )
                    + "\n"
                )

        LOGGER.info(
            "[manifests] Auto-generated %d frame records into %s",
            len(frames_records),
            frames_jsonl,
        )
    else:
        LOGGER.warning(
            "[manifests] Could not auto-generate manifests (no frame_ids*.json or frames/ directory found)."
        )


def events_to_query(payload: dict) -> str:
    """Build a ``trake_search`` query string from a TRAKE payload."""
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        return str(payload.get("query", ""))

    lines = []
    for index, event in enumerate(events, start=1):
        if isinstance(event, dict):
            text = str(event.get("text", "")).strip()
            details = [str(d).strip() for d in event.get("sub_details") or [] if str(d).strip()]
        else:
            text = str(event).strip()
            details = []
        if not text:
            continue
        if details:
            text = f"{text} {' '.join(details)}"
        lines.append(f"E{index}: {text}")
    return "\n".join(lines)


_VIDEO_NAME_CACHE: dict[str, dict[str, str]] = {}
_CAPTIONS_CACHE: dict[str, str] = {}
_TEXT_CACHE: list[dict] = []


def _get_video_name_map(experiment: Experiment) -> dict[str, str]:
    global _VIDEO_NAME_CACHE
    cache_key = str(experiment.run_dir.resolve())
    if cache_key not in _VIDEO_NAME_CACHE:
        names: dict[str, str] = {}
        v_file = experiment.run_dir / "manifests" / "videos.jsonl"
        if v_file.is_file():
            try:
                with open(v_file, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        vid = rec.get("video_id")
                        vpath = rec.get("path")
                        if vid and vpath:
                            normalized = str(vpath).replace("\\", "/")
                            names[str(vid)] = normalized.rsplit("/", 1)[-1]
            except Exception:
                pass
        _VIDEO_NAME_CACHE[cache_key] = names
    return _VIDEO_NAME_CACHE[cache_key]


def _get_captions_map(experiment: Experiment) -> dict[str, str]:
    global _CAPTIONS_CACHE
    if not _CAPTIONS_CACHE:
        c_file = experiment.run_dir / "manifests" / "captions.jsonl"
        if c_file.is_file():
            try:
                with open(c_file, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        fid = rec.get("frame_id")
                        cap = rec.get("caption")
                        if fid and cap:
                            _CAPTIONS_CACHE[fid] = cap
            except Exception:
                pass
    return _CAPTIONS_CACHE


def _get_text_records(experiment: Experiment) -> list[dict]:
    global _TEXT_CACHE
    if not _TEXT_CACHE:
        t_file = experiment.run_dir / "manifests" / "text.jsonl"
        if t_file.is_file():
            try:
                with open(t_file, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        _TEXT_CACHE.append(json.loads(line))
            except Exception:
                pass
    return _TEXT_CACHE


def result_to_payload(
    result: SearchResult, experiment: Experiment | None = None
) -> dict[str, object]:
    """Serialize a result for the browser UI."""
    payload = result.to_dict()
    if result.frame_path:
        payload["image_url"] = f"/frame?path={quote(result.frame_path)}"

    if experiment:
        vmap = _get_video_name_map(experiment)
        if result.video_id in vmap:
            payload["video_name"] = vmap[result.video_id]
        elif not payload.get("video_name"):
            payload["video_name"] = result.video_id
    elif not payload.get("video_name"):
        payload["video_name"] = result.video_id

    return payload


def handle_trake_or_enhanced_search(
    path: str,
    payload: dict,
    experiment: Experiment,
    default_top_k: int,
    reranker=None,
    reranker_top_k: int = 10,
) -> dict:
    """Process /api/trake-search and /api/enhanced-temporal-search."""
    top_k = int(payload.get("top_k") or default_top_k)
    window = int(payload.get("window", 15))
    events_raw = payload.get("events")
    if path == "/api/trake-search" and isinstance(events_raw, list) and len(events_raw) >= 2:
        from retrieval.trake_search import trake_bpj_search

        enabled_models = payload.get("enabled_models") or None
        use_reranker = payload.get("use_reranker")
        if use_reranker is not None:
            use_reranker = bool(use_reranker)
        use_llm = payload.get("use_llm")
        if use_llm is not None:
            use_llm = bool(use_llm)

        result = trake_bpj_search(
            experiment=experiment,
            events=events_raw,
            top_k=300,
            window=window,
            enabled_models=enabled_models,
            use_reranker=use_reranker,
            use_llm=use_llm,
        )
    else:
        req_reranker_top_k = payload.get("reranker_top_k")
        req_reranker_top_k = int(req_reranker_top_k) if req_reranker_top_k else None
        shared = {
            "experiment": experiment,
            "context": str(payload.get("context", "")),
            "top_k": top_k,
            "reranker": reranker if req_reranker_top_k else None,
            "reranker_top_k": req_reranker_top_k or reranker_top_k,
        }
        if path == "/api/enhanced-temporal-search":
            result = enhanced_temporal_search(
                query=str(payload.get("query", "")),
                max_events=int(payload.get("max_events") or 5),
                **shared,
            )
        else:
            result = trake_search(query=events_to_query(payload), **shared)

    for video in result.get("videos", []):
        for ev in video.get("events", []):
            if ev.get("frame_path"):
                ev["image_url"] = f"/frame?path={quote(ev['frame_path'])}"
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    for ev in result.get("events", []):
        ev["image_urls"] = [f"/frame?path={quote(fp)}" for fp in ev.get("frame_paths", []) if fp]
    return result


def handle_vqa_search(
    payload: dict,
    experiment: Experiment,
    default_top_k: int,
    reranker=None,
    reranker_top_k: int = 10,
) -> dict:
    """Process /api/vqa-search."""
    top_k = int(payload.get("top_k") or default_top_k)
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    req_reranker_top_k = payload.get("reranker_top_k")
    req_reranker_top_k = int(req_reranker_top_k) if req_reranker_top_k else None
    if req_reranker_top_k is not None and not 1 <= req_reranker_top_k <= 100:
        raise ValueError("reranker_top_k must be between 1 and 100")
    vqa_backend = payload.get("vqa_backend", "local")
    enabled_models = payload.get("enabled_models")
    if enabled_models is not None and not isinstance(enabled_models, list):
        raise ValueError("enabled_models must be an array")
    if isinstance(enabled_models, list):
        enabled_models = list(
            dict.fromkeys(str(model).strip() for model in enabled_models if str(model).strip())
        )
        if not enabled_models:
            raise ValueError("enabled_models must contain at least one model")
    use_reranker = _parse_json_bool(
        payload.get("use_reranker"), field="use_reranker", default=None
    )
    use_llm = _parse_json_bool(
        payload.get("use_llm"), field="use_llm", default=True
    )
    pipeline_mode = str(payload.get("pipeline_mode") or "grounded").strip().lower()
    if pipeline_mode not in {"grounded", "legacy"}:
        raise ValueError("pipeline_mode must be 'grounded' or 'legacy'")
    reranker_requested = (
        use_reranker if use_reranker is not None else req_reranker_top_k is not None
    )

    result = vqa_search(
        experiment=experiment,
        query=str(payload.get("query", "")),
        question=str(payload.get("question", "")),
        context=str(payload.get("context", "")),
        top_k=top_k,
        reranker=reranker if reranker_requested else None,
        reranker_top_k=req_reranker_top_k or reranker_top_k,
        vqa_backend=vqa_backend,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
        pipeline_mode=pipeline_mode,
    )
    _attach_nested_frame_image_urls(result)
    return result


def handle_agent_chat(payload: dict, experiment: Experiment) -> dict:
    """Process /api/agent/chat."""
    from agent.interactive import run_agent_turn

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages (non-empty list) is required.")
    result = run_agent_turn(messages, experiment)
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    return result


def handle_text_search(
    path: str, payload: dict, experiment: Experiment, default_top_k: int
) -> dict:
    """Process /api/asr-search and /api/ocr-search."""
    query = str(payload.get("query", ""))
    top_k = int(payload.get("top_k") or default_top_k)
    source = "asr" if path == "/api/asr-search" else "ocr"

    result = text_search(experiment, query=query, source=source, top_k=top_k)
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    return result


def handle_intelligent_search(payload: dict, experiment: Experiment, default_top_k: int) -> dict:
    """Process /api/intelligent-search."""
    enabled_models = payload.get("enabled_models") or None
    use_reranker = payload.get("use_reranker")
    if use_reranker is not None:
        use_reranker = bool(use_reranker)
    use_llm = payload.get("use_llm")
    use_llm = True if use_llm is None else bool(use_llm)
    use_evidence_reranker = payload.get("use_evidence_reranker")
    if use_evidence_reranker is not None:
        use_evidence_reranker = bool(use_evidence_reranker)

    result = intelligent_search(
        experiment,
        query=str(payload.get("query", "")),
        top_k=int(payload.get("top_k") or default_top_k),
        enable_kis=bool(payload.get("enable_kis", True)),
        enable_ocr=bool(payload.get("enable_ocr", True)),
        enable_asr=bool(payload.get("enable_asr", True)),
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
        fusion_mode=str(payload.get("fusion_mode", "adaptive")),
        text_search_mode=str(payload.get("text_search_mode", "separate")),
        temporal_asr=bool(payload.get("temporal_asr", True)),
        use_evidence_reranker=use_evidence_reranker,
        max_frames_per_shot=int(payload.get("max_frames_per_shot", 2)),
    )
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    return result


def handle_kis_detail_2stage(payload: dict, experiment: Experiment) -> dict:
    """Process /api/kis-detail-2stage."""
    general_raw = payload.get("general")
    specific_raw = payload.get("specific")
    if not isinstance(general_raw, list) or len(general_raw) < 1:
        raise ValueError("At least 1 general subquery is required.")
    if not isinstance(specific_raw, list) or len(specific_raw) < 1:
        raise ValueError("At least 1 specific subquery is required.")
    general = [str(s).strip() for s in general_raw if str(s).strip()]
    specific = [str(s).strip() for s in specific_raw if str(s).strip()]

    enabled_models = payload.get("enabled_models") or None

    result = kis_detail_2stage_search(
        experiment=experiment,
        general=general,
        specific=specific,
        general_weights=payload.get("general_weights"),
        specific_weights=payload.get("specific_weights"),
        enabled_models=enabled_models,
    )
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    return result


def handle_compute_sub_score(
    payload: dict, experiment: Experiment, retriever
) -> tuple[dict, HTTPStatus]:
    """Process /api/compute-sub-score."""
    frame_id = payload.get("frame_id")
    sub_text = payload.get("sub_text")
    if not frame_id or not sub_text:
        raise ValueError("frame_id and sub_text are required.")

    model_name = next(iter(retriever.embedders))
    frame_vec = retriever.index.get_vector(frame_id, model_name)
    if frame_vec is None:
        return {"error": "frame_id not found"}, HTTPStatus.NOT_FOUND

    frame_vec = np.asarray(frame_vec, dtype="float32")
    sub_vec = np.asarray(
        retriever.embedders[model_name].embed_text(sub_text), dtype="float32"
    ).flatten()
    nrm = np.linalg.norm(sub_vec)
    if nrm > 1e-12:
        sub_vec /= nrm
    score = float(frame_vec @ sub_vec)
    return {"score": round(score, 4)}, HTTPStatus.OK


def handle_default_search(
    payload: dict, experiment: Experiment, retriever, default_top_k: int
) -> dict:
    """Process /api/search."""
    request = TrackQuery(
        track=str(payload.get("track", "textual_kis")),
        query=str(payload.get("query", "")),
        question=str(payload.get("question", "")),
        context=str(payload.get("context", "")),
    )
    top_k = int(payload.get("top_k") or default_top_k)
    enabled_models = payload.get("enabled_models") or None
    use_reranker = payload.get("use_reranker")
    if use_reranker is not None:
        use_reranker = bool(use_reranker)
    use_llm = payload.get("use_llm")
    if use_llm is not None:
        use_llm = bool(use_llm)

    retrieval_text = build_retrieval_text(request)
    results = retriever.search(
        query=retrieval_text,
        top_k=top_k,
        enabled_models=enabled_models,
        use_reranker=use_reranker,
        use_llm=use_llm,
    )

    return {
        "track": request.track,
        "track_label": SUPPORTED_TRACKS.get(request.track, request.track),
        "retrieval_text": retrieval_text,
        "results": [result_to_payload(result, experiment) for result in results],
    }


_FRAMES_BY_VIDEO_CACHE: dict[str, tuple[list[str], dict[str, list[dict]]]] = {}
_VIDEO_TEXT_CACHE: dict[str, tuple[list[dict], list[dict]]] = {}


def _get_frames_by_video(
    experiment: Experiment, video_id: str
) -> tuple[list[str], dict[str, list[dict]]]:
    global _FRAMES_BY_VIDEO_CACHE
    if not _FRAMES_BY_VIDEO_CACHE:
        frames_path = experiment.run_dir / "manifests" / "frames.jsonl"
        if frames_path.is_file():
            try:
                with open(frames_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        frame = json.loads(line)
                        vid = frame.get("video_id")
                        if not vid:
                            continue
                        sid = frame.get("shot_id") or "s0"

                        if vid not in _FRAMES_BY_VIDEO_CACHE:
                            _FRAMES_BY_VIDEO_CACHE[vid] = ([], {})

                        shot_order, frames_by_shot = _FRAMES_BY_VIDEO_CACHE[vid]
                        if sid not in frames_by_shot:
                            frames_by_shot[sid] = []
                            shot_order.append(sid)

                        ts = frame.get("timestamp_sec")
                        idx = frame.get("frame_index")
                        fid = frame.get("frame_id", "")
                        match = re.search(r"_f(\d+)", fid)
                        if match:
                            f_num = int(match.group(1))
                            if idx is None or idx == 0:
                                idx = f_num

                        frame_copy = dict(frame)
                        frame_copy["timestamp_sec"] = ts if ts is not None else 0.0
                        frame_copy["frame_index"] = idx if idx is not None else 0
                        frames_by_shot[sid].append(frame_copy)
            except Exception:
                LOGGER.exception("Failed to build frames by video cache")

    return _FRAMES_BY_VIDEO_CACHE.get(video_id, ([], {}))


def _get_video_text_texts(experiment: Experiment, video_id: str) -> tuple[list[dict], list[dict]]:
    global _VIDEO_TEXT_CACHE
    if not _VIDEO_TEXT_CACHE:
        text_records = _get_text_records(experiment)
        for txt in text_records:
            vid = txt.get("video_id")
            if not vid:
                continue
            if vid not in _VIDEO_TEXT_CACHE:
                _VIDEO_TEXT_CACHE[vid] = ([], [])
            src = txt.get("source", "asr")
            rec = {
                "doc_id": txt.get("doc_id"),
                "video_id": vid,
                "text": txt.get("text"),
                "timestamp_sec": txt.get("timestamp_sec", 0.0),
                "frame_id": txt.get("frame_id"),
            }
            if src == "ocr":
                _VIDEO_TEXT_CACHE[vid][1].append(rec)
            elif src == "asr":
                _VIDEO_TEXT_CACHE[vid][0].append(rec)
        temporal_mapper = AsrTemporalMapper(experiment)
        for asr, _ in _VIDEO_TEXT_CACHE.values():
            for record in asr:
                interval = temporal_mapper.interval_for(record)
                record["start_time_sec"], record["end_time_sec"] = interval
                record["mapped_frame_ids"] = {
                    frame_id for frame_id, _ in temporal_mapper.map_document(record)
                }
    return _VIDEO_TEXT_CACHE.get(video_id, ([], []))


def handle_video_shots(query: dict, experiment: Experiment) -> tuple[dict, HTTPStatus]:
    """Process /api/video-shots GET endpoint."""
    video_id = query.get("video_id", [""])[0]
    if not video_id:
        return {"error": "video_id required"}, HTTPStatus.BAD_REQUEST

    vmap = _get_video_name_map(experiment)
    video_name = vmap.get(video_id, video_id)

    captions_map = _get_captions_map(experiment)
    shot_order, frames_by_shot = _get_frames_by_video(experiment, video_id)
    video_asr_texts, video_ocr_texts = _get_video_text_texts(experiment, video_id)

    try:
        shot_list = []
        for sid in shot_order:
            shot_frames = frames_by_shot[sid]

            formatted_frames = []
            for f in shot_frames:
                fid = f.get("frame_id", "")
                fts = f.get("timestamp_sec", 0.0)

                matched_asr = [
                    a["text"]
                    for a in video_asr_texts
                    if a.get("frame_id") == fid
                    or (not a.get("frame_id") and fid in a.get("mapped_frame_ids", set()))
                ]
                matched_ocr = [
                    o["text"]
                    for o in video_ocr_texts
                    if o.get("frame_id") == fid
                    or (not o.get("frame_id") and o.get("timestamp_sec") == fts)
                ]

                formatted_frames.append(
                    {
                        "frame_id": fid,
                        "frame_index": f.get("frame_index", 0),
                        "timestamp_sec": fts,
                        "frame_path": f.get("frame_path"),
                        "image_url": f"/frame?path={quote(f.get('frame_path', ''))}",
                        "caption": captions_map.get(fid, ""),
                        "asr": " | ".join(matched_asr) if matched_asr else "",
                        "ocr": " | ".join(matched_ocr) if matched_ocr else "",
                    }
                )

            shot_data = {
                "shot_id": sid,
                "start_frame": shot_frames[0].get("frame_index", 0),
                "end_frame": shot_frames[-1].get("frame_index", 0),
                "start_time_sec": shot_frames[0].get("timestamp_sec", 0.0),
                "end_time_sec": shot_frames[-1].get("timestamp_sec", 0.0),
                "frames": formatted_frames,
            }
            shot_list.append(shot_data)

        return {"video_id": video_id, "video_name": video_name, "shots": shot_list}, HTTPStatus.OK
    except Exception:
        LOGGER.exception("Failed to load shots for video=%s", video_id)
        return {"video_id": video_id, "video_name": video_name, "shots": []}, HTTPStatus.OK
