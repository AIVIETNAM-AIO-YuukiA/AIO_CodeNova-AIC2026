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
from retrieval.temporal_search import load_temporal_data
from retrieval.text_search import text_search
from retrieval.tracks import SUPPORTED_TRACKS, TrackQuery, build_retrieval_text
from retrieval.vqa import enhanced_temporal_search, trake_search, vqa_search
import numpy as np

LOGGER = get_logger(__name__)


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
    json_ids_files = (
        list(embeddings_dir.glob("frame_ids*.json")) if embeddings_dir.exists() else []
    )

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
                ts = round(f_num / 25.0, 2) if f_num > 0 else 0.0

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
                ts = round(f_num / 25.0, 2) if f_num > 0 else 0.0

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


def result_to_payload(result: SearchResult) -> dict[str, object]:
    """Serialize a result for the browser UI."""
    payload = result.to_dict()
    if result.frame_path:
        payload["image_url"] = f"/frame?path={quote(result.frame_path)}"
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
        ev["image_urls"] = [
            f"/frame?path={quote(fp)}" for fp in ev.get("frame_paths", []) if fp
        ]
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
    req_reranker_top_k = payload.get("reranker_top_k")
    req_reranker_top_k = int(req_reranker_top_k) if req_reranker_top_k else None
    vqa_backend = payload.get("vqa_backend", "local")

    result = vqa_search(
        experiment=experiment,
        query=str(payload.get("query", "")),
        question=str(payload.get("question", "")),
        context=str(payload.get("context", "")),
        top_k=top_k,
        reranker=reranker if req_reranker_top_k else None,
        reranker_top_k=req_reranker_top_k or reranker_top_k,
        vqa_backend=vqa_backend,
    )
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
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
    result = intelligent_search(
        experiment,
        query=str(payload.get("query", "")),
        top_k=int(payload.get("top_k") or default_top_k),
        enable_kis=bool(payload.get("enable_kis", True)),
        enable_ocr=bool(payload.get("enable_ocr", True)),
        enable_asr=bool(payload.get("enable_asr", True)),
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
    result = kis_detail_2stage_search(
        experiment=experiment,
        general=general,
        specific=specific,
        general_weights=payload.get("general_weights"),
        specific_weights=payload.get("specific_weights"),
    )
    for r in result.get("results", []):
        if r.get("frame_path"):
            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
    return result


def handle_compute_sub_score(payload: dict, experiment: Experiment, retriever) -> tuple[dict, HTTPStatus]:
    """Process /api/compute-sub-score."""
    frame_id = payload.get("frame_id")
    sub_text = payload.get("sub_text")
    if not frame_id or not sub_text:
        raise ValueError("frame_id and sub_text are required.")

    frame_embeddings, frame_records = load_temporal_data(experiment)
    idx = None
    for i, rec in enumerate(frame_records):
        if rec.get("frame_id") == frame_id:
            idx = i
            break
    if idx is None:
        return {"error": "frame_id not found"}, HTTPStatus.NOT_FOUND

    frame_vec = frame_embeddings[idx]
    sub_vec = np.asarray(retriever.embedder.embed_text(sub_text), dtype="float32").flatten()
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
        "results": [result_to_payload(result) for result in results],
    }


def handle_video_shots(query: dict, experiment: Experiment) -> tuple[dict, HTTPStatus]:
    """Process /api/video-shots GET endpoint."""
    video_id = query.get("video_id", [""])[0]
    if not video_id:
        return {"error": "video_id required"}, HTTPStatus.BAD_REQUEST

    frames_path = experiment.run_dir / "manifests" / "frames.jsonl"
    experiment.run_dir / "manifests" / "shots.jsonl"

    try:
        frames_by_shot: dict[str, list[dict]] = {}
        shot_order: list[str] = []

        if frames_path.is_file():
            with open(frames_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    frame = json.loads(line)
                    if frame.get("video_id") == video_id:
                        sid = frame.get("shot_id") or "s0"
                        if sid not in frames_by_shot:
                            frames_by_shot[sid] = []
                            shot_order.append(sid)

                        ts = frame.get("timestamp_sec")
                        idx = frame.get("frame_index")
                        fid = frame.get("frame_id", "")
                        match = re.search(r"_f(\d+)", fid)
                        if match:
                            f_num = int(match.group(1))
                            ts = round(f_num / 25.0, 2)
                            if idx is None or idx == 0:
                                idx = f_num

                        frame_copy = dict(frame)
                        frame_copy["timestamp_sec"] = ts if ts is not None else 0.0
                        frame_copy["frame_index"] = idx if idx is not None else 0
                        frames_by_shot[sid].append(frame_copy)

        shot_list = []
        for sid in shot_order:
            shot_frames = frames_by_shot[sid]
            shot_frames.sort(key=lambda x: x.get("frame_index", 0))

            shot_data = {
                "shot_id": sid,
                "start_frame": shot_frames[0].get("frame_index", 0),
                "end_frame": shot_frames[-1].get("frame_index", 0),
                "start_time_sec": shot_frames[0].get("timestamp_sec", 0.0),
                "end_time_sec": shot_frames[-1].get("timestamp_sec", 0.0),
                "frames": [
                    {
                        "frame_id": f.get("frame_id"),
                        "frame_index": f.get("frame_index", 0),
                        "timestamp_sec": f.get("timestamp_sec", 0.0),
                        "frame_path": f.get("frame_path"),
                        "image_url": f"/frame?path={quote(f.get('frame_path', ''))}",
                    }
                for f in shot_frames
            ],
        }
            shot_list.append(shot_data)

        return {"video_id": video_id, "shots": shot_list}, HTTPStatus.OK
    except Exception:
        LOGGER.exception("Failed to load shots for video=%s", video_id)
        return {"video_id": video_id, "shots": []}, HTTPStatus.OK
