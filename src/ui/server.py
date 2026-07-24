"""Small stdlib web UI for query and result image inspection."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
import json
from core.logging import get_logger
import mimetypes

from config.settings import Experiment
from core.types import SearchResult
from retrieval.vqa import vqa_search, trake_search
from retrieval.kis_detail_search import kis_detail_search
from retrieval import build_retriever
from retrieval.temporal_search import load_temporal_data
from retrieval.tracks import SUPPORTED_TRACKS, TrackQuery, build_retrieval_text
import numpy as np

LOGGER = get_logger(__name__)


def _warmup_models(reranker, experiment: Experiment) -> None:
    """Pre-load heavy models in background so the first request doesn't block.

    Called in a daemon thread right after the server starts listening.
    The lazy-loaded models (BLIP-2 reranker, SigLIP embedder) can take
    3-4 minutes to download/load on Colab T4. By triggering loading here,
    models are ready before the user makes their first request.
    """
    import threading

    def _load():
        LOGGER.info("[warmup] Starting background model pre-loading...")
        try:
            # Pre-load the reranker (BLIP-2) if one is configured.
            if reranker is not None and hasattr(reranker, "_load"):
                LOGGER.info("[warmup] Loading BLIP-2 reranker...")
                reranker._load()
                LOGGER.info("[warmup] BLIP-2 reranker ready.")
        except Exception as exc:
            LOGGER.warning("[warmup] Reranker pre-load failed (non-fatal): %s", exc)

        try:
            # Pre-load the embedder (SigLIP) — also used by the retriever.
            from modules.embedding import build_embedder

            embedder = build_embedder(
                model_name=experiment.config.embedding_model,
                device=experiment.config.device,
            )
            embedder.embed_text("warmup query")
            LOGGER.info("[warmup] Embedder ready.")
        except Exception as exc:
            LOGGER.warning("[warmup] Embedder pre-load failed (non-fatal): %s", exc)

        LOGGER.info("[warmup] All models pre-loaded and ready.")

    t = threading.Thread(target=_load, daemon=True, name="model-warmup")
    t.start()


def serve_ui(
    experiment: Experiment,
    host: str = "127.0.0.1",
    port: int = 7860,
    default_top_k: int = 20,
    reranker=None,
    reranker_top_k: int = 10,
) -> None:
    """Serve the local retrieval UI until interrupted."""
    from core.paths import set_project_root

    # experiment.run_dir is typically runs/<experiment_name>
    # so its parent is the runs/ directory, and its parent is the project root
    set_project_root(experiment.run_dir.parent.parent)

    retriever = build_retriever(experiment)
    handler = build_handler(
        experiment=experiment,
        retriever=retriever,
        default_top_k=default_top_k,
        reranker=reranker,
        reranker_top_k=reranker_top_k,
    )
    server = ThreadingHTTPServer((host, port), handler)
    LOGGER.info("Serving retrieval UI at http://%s:%s", host, port)
    # Kick off background model loading immediately so the first user request
    # doesn't have to wait 3-4 minutes for BLIP-2 / SigLIP to load.
    _warmup_models(reranker, experiment)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping retrieval UI")
    finally:
        server.server_close()


def build_handler(
    experiment: Experiment, retriever, default_top_k: int, reranker=None, reranker_top_k: int = 10
):
    """Create a request handler bound to one experiment and its retriever."""

    class RetrievalUiHandler(BaseHTTPRequestHandler):
        server_version = "CodeNovaRetrievalUI/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html = INDEX_HTML.replace('value="20"', f'value="{default_top_k}"')
                self._send_html(html)
                return
            if parsed.path == "/health":
                self._send_json({"ok": True, "experiment": experiment.name})
                return
            if parsed.path == "/frame":
                self._send_frame(parse_qs(parsed.query).get("path", [""])[0])
                return
            if parsed.path == "/api/video-shots":
                self._send_video_shots(parse_qs(parsed.query))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)

            # TRAKE track: Embedding search → Rerank (optional) → Temporal → event segments
            if parsed.path == "/api/trake-search":
                try:
                    payload = self._read_json()
                    top_k = int(payload.get("top_k") or default_top_k)
                    req_reranker_top_k = payload.get("reranker_top_k")
                    req_reranker_top_k = int(req_reranker_top_k) if req_reranker_top_k else None

                    result = trake_search(
                        experiment=experiment,
                        query=str(payload.get("query", "")),
                        context=str(payload.get("context", "")),
                        top_k=top_k,
                        reranker=reranker if req_reranker_top_k else None,
                        reranker_top_k=req_reranker_top_k or reranker_top_k,
                    )
                    for r in result.get("results", []):
                        if r.get("frame_path"):
                            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
                    for ev in result.get("events", []):
                        ev["image_urls"] = [
                            f"/frame?path={quote(fp)}" for fp in ev.get("frame_paths", []) if fp
                        ]
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("KIS Detail search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # VQA track uses full pipeline (Embedding search → Temporal → Agent)
            if parsed.path == "/api/vqa-search":
                try:
                    payload = self._read_json()
                    top_k = int(payload.get("top_k") or default_top_k)
                    req_reranker_top_k = payload.get("reranker_top_k")
                    req_reranker_top_k = int(req_reranker_top_k) if req_reranker_top_k else None
                    vqa_backend = payload.get("vqa_backend", "gemini")

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
                    # Hydrate frame paths for image serving
                    for r in result.get("results", []):
                        if r.get("frame_path"):
                            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("VQA search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # KIS Detail: in-memory multi-concept sum fusion (independent of TRAKE)
            if parsed.path == "/api/kis-detail":
                try:
                    payload = self._read_json()
                    subqueries_raw = payload.get("subqueries")
                    if not isinstance(subqueries_raw, list) or len(subqueries_raw) < 1:
                        raise ValueError("At least 1 subquery is required.")
                    subqueries = [str(s).strip() for s in subqueries_raw if str(s).strip()]
                    result = kis_detail_search(
                        experiment=experiment,
                        subqueries=subqueries,
                        top_k=300,
                    )
                    for r in result.get("results", []):
                        if r.get("frame_path"):
                            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("KIS Detail search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # Compute sub-detail score for a specific frame (live)
            if parsed.path == "/api/compute-sub-score":
                try:
                    payload = self._read_json()
                    frame_id = payload.get("frame_id")
                    sub_text = payload.get("sub_text")
                    if not frame_id or not sub_text:
                        raise ValueError("frame_id and sub_text are required.")

                    frame_embeddings, frame_records = load_temporal_data(experiment.run_dir)
                    idx = None
                    for i, rec in enumerate(frame_records):
                        if rec.get("frame_id") == frame_id:
                            idx = i
                            break
                    if idx is None:
                        self._send_json(
                            {"error": "frame_id not found"}, status=HTTPStatus.NOT_FOUND
                        )
                        return

                    frame_vec = frame_embeddings[idx]
                    sub_vec = np.asarray(
                        retriever.embedder.embed_text(sub_text), dtype="float32"
                    ).flatten()
                    nrm = np.linalg.norm(sub_vec)
                    if nrm > 1e-12:
                        sub_vec /= nrm
                    score = float(frame_vec @ sub_vec)
                    self._send_json({"score": round(score, 4)})
                except Exception as exc:
                    LOGGER.exception("compute-sub-score failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path != "/api/search":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            try:
                payload = self._read_json()
                request = TrackQuery(
                    track=str(payload.get("track", "textual_kis")),
                    query=str(payload.get("query", "")),
                    question=str(payload.get("question", "")),
                    context=str(payload.get("context", "")),
                )
                top_k = int(payload.get("top_k") or default_top_k)
                req_reranker_top_k = payload.get("reranker_top_k")
                req_reranker_top_k = int(req_reranker_top_k) if req_reranker_top_k else None

                retrieval_text = build_retrieval_text(request)
                results = retriever.search(query=retrieval_text, top_k=top_k)

                if reranker and req_reranker_top_k:
                    results = reranker.rerank(query=retrieval_text, results=results)[
                        :req_reranker_top_k
                    ]

                self._send_json(
                    {
                        "track": request.track,
                        "track_label": SUPPORTED_TRACKS.get(request.track, request.track),
                        "retrieval_text": retrieval_text,
                        "results": [result_to_payload(result) for result in results],
                    }
                )
            except Exception as exc:
                LOGGER.exception("UI search failed")
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.debug("ui %s", format % args)

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))

        def _send_html(self, html: str) -> None:
            encoded = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(
            self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            encoded = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_frame(self, raw_path: str) -> None:
            raw_path = unquote(raw_path).replace("\\", "/")
            frame_path = Path(raw_path).resolve()
            frames_root = (experiment.run_dir / "frames").resolve()
            if not frame_path.is_file() or not frame_path.is_relative_to(frames_root):
                self.send_error(HTTPStatus.NOT_FOUND, "Frame not found")
                return

            content_type = mimetypes.guess_type(frame_path.name)[0] or "application/octet-stream"
            content = frame_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_video_shots(self, query: dict) -> None:
            video_id = query.get("video_id", [""])[0]
            if not video_id:
                self._send_json({"error": "video_id required"}, status=HTTPStatus.BAD_REQUEST)
                return

            shots_path = experiment.run_dir / "manifests" / "shots.jsonl"
            frames_path = experiment.run_dir / "manifests" / "frames.jsonl"

            try:
                # Parse shots for this video
                shots = []
                with open(shots_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        shot = json.loads(line)
                        if shot.get("video_id") == video_id:
                            shots.append(shot)

                # Parse frames for this video, grouped by shot_id
                frames_by_shot: dict[str, list[dict]] = {}
                with open(frames_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        frame = json.loads(line)
                        if frame.get("video_id") == video_id:
                            sid = frame.get("shot_id")
                            if sid:
                                frames_by_shot.setdefault(sid, []).append(frame)

                shot_list = []
                for shot in shots:
                    sid = shot.get("shot_id")
                    shot_frames = frames_by_shot.get(sid, [])
                    shot_frames.sort(key=lambda x: x.get("frame_index", 0))

                    # Pick 3 frames evenly distributed (or fewer if less available)
                    if len(shot_frames) > 3:
                        indices = [0, len(shot_frames) // 2, len(shot_frames) - 1]
                        picked = [shot_frames[i] for i in indices]
                    else:
                        picked = shot_frames

                    shot_data = {
                        "shot_id": sid,
                        "start_frame": shot.get("start_frame"),
                        "end_frame": shot.get("end_frame"),
                        "start_time_sec": shot.get("start_time_sec"),
                        "end_time_sec": shot.get("end_time_sec"),
                        "frames": [
                            {
                                "frame_id": f.get("frame_id"),
                                "frame_index": f.get("frame_index"),
                                "timestamp_sec": f.get("timestamp_sec"),
                                "frame_path": f.get("frame_path"),
                                "image_url": f"/frame?path={quote(f.get('frame_path', ''))}",
                            }
                            for f in picked
                        ],
                    }
                    shot_list.append(shot_data)

                self._send_json({"video_id": video_id, "shots": shot_list})
            except Exception:
                LOGGER.exception("Failed to load shots for video=%s", video_id)
                self._send_json({"video_id": video_id, "shots": []})

    return RetrievalUiHandler


def result_to_payload(result: SearchResult) -> dict[str, object]:
    """Serialize a result for the browser UI."""
    payload = result.to_dict()
    if result.frame_path:
        payload["image_url"] = f"/frame?path={quote(result.frame_path)}"
    return payload


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CodeNova Retrieval UI</title>
  <style>
    :root {
      --bg: #f7f7f4; --panel: #ffffff; --text: #1c1f24;
      --muted: #667085; --line: #d9dde3;
      --accent: #0f766e; --accent-strong: #115e59; --warn: #a16207;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: var(--text); background: var(--bg); }
    header { padding: 18px 24px 12px; border-bottom: 1px solid var(--line); background: var(--panel); }
    h1 { margin: 0; font-size: 20px; }
    main { display: grid; grid-template-columns: minmax(320px, 420px) 1fr; min-height: calc(100vh - 61px); }
    aside { padding: 18px; border-right: 1px solid var(--line); background: var(--panel); }
    section { padding: 18px; }
    label {
      display: block;
      margin: 14px 0 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    select, input, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
    }
    select, input, textarea {
      padding: 10px 11px;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 96px;
      resize: vertical;
      line-height: 1.45;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      align-items: end;
    }
    button {
      margin-top: 16px;
      padding: 11px 14px;
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
      font-weight: 750;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: .65; cursor: wait; }
    .hint, .status { margin-top: 12px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .status strong { color: var(--text); }
    .status.warn { color: var(--warn); }
    .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; background: #e6f5f3; color: var(--accent-strong); font-size: 12px; font-weight: 700; }
    /* Results grid */
    .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; }
    .card { overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    .card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; background: #e5e7eb; cursor: zoom-in; }
    .meta { padding: 10px 11px 12px; font-size: 13px; line-height: 1.45; }
    .meta code { display: block; overflow-wrap: anywhere; margin-top: 4px; color: var(--muted); font-size: 12px; }
    /* Answer / pipeline */
    .answer-box { margin-bottom: 18px; padding: 18px 20px; border: 2px solid var(--accent); border-radius: 10px; background: #f0fdf8; }
    .answer-box .label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
    .answer-box .answer-text { margin-top: 6px; font-size: 18px; font-weight: 700; color: var(--accent-strong); line-height: 1.45; }
    .pipeline-toggle { margin-top: 10px; background: none; border: 1px solid var(--line); padding: 6px 12px; border-radius: 6px; color: var(--muted); cursor: pointer; font-size: 12px; }
    .pipeline-toggle:hover { background: var(--panel); }
    .pipeline-detail { display: none; margin-top: 10px; padding: 14px 16px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); font-size: 13px; line-height: 1.5; }
    .pipeline-detail.open { display: block; }
    .pipeline-detail code { display: block; white-space: pre-wrap; font-size: 12px; color: var(--muted); }
    /* TRAKE event card */
    .video-block { margin-bottom: 20px; padding: 14px 16px; border: 2px solid var(--accent); border-radius: 10px; background: var(--panel); }
    .video-block-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .event-grid { display: grid; gap: 10px; }
    /* Each event card */
    .ev-card { position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); background: #f0fdf8; }
    .ev-card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; background: #e5e7eb; cursor: zoom-in; transition: opacity .15s; }
    .ev-card img:hover { opacity: .88; }
    .ev-card .ev-info { padding: 5px 8px 6px; font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; gap: 6px; }
    /* Revert badge — shown only when thumbnail has been changed */
    .ev-card .revert-badge {
      display: none; position: absolute; top: 5px; right: 5px;
      background: rgba(0,0,0,.65); color: #fff; border: none;
      border-radius: 5px; padding: 3px 8px; font-size: 11px; font-weight: 600;
      cursor: pointer; margin-top: 0; width: auto;
    }
    .ev-card .revert-badge:hover { background: rgba(0,0,0,.85); }
    .ev-card.has-custom .revert-badge { display: block; }
    .ev-card.has-custom { border-color: var(--accent); }
    /* Modal */
    #frame-modal {
      display: none; position: fixed; inset: 0; z-index: 999;
      background: rgba(0,0,0,.88); align-items: center; justify-content: center;
    }
    #frame-modal.open { display: flex; }
    .modal-box {
      display: flex; flex-direction: column;
      width: 95vw; height: 95vh; max-width: 1400px;
      border-radius: 12px; overflow: hidden;
      background: #1a1a1a; box-shadow: 0 8px 40px rgba(0,0,0,.6);
    }
    .modal-top {
      display: flex; justify-content: space-between; align-items: center;
      padding: 8px 14px; background: #222; color: #eee; font-size: 13px; flex-shrink: 0;
    }
    .modal-top .time-badge { color: #0f766e; font-weight: 600; }
    .modal-top .close-x { background: none; border: none; color: #aaa; font-size: 22px; cursor: pointer; padding: 0 4px; margin-top: 0; width: auto; }
    .modal-top .close-x:hover { color: #fff; background: none; }
    .modal-mid { flex: 1; display: flex; align-items: stretch; padding: 6px; gap: 6px; min-height: 0; }
    .modal-mid .img-area { flex: 1; display: flex; justify-content: center; align-items: center; min-height: 0; overflow: hidden; }
    .modal-mid .img-area img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 6px; display: block; }
    .modal-nav { flex-shrink: 0; background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.2); color: #fff; font-size: 22px; cursor: pointer; padding: 0 14px; border-radius: 6px; display: flex; align-items: center; justify-content: center; margin-top: 0; width: auto; }
    .modal-nav:hover { background: rgba(255,255,255,.25); }
    .modal-nav:disabled { opacity: .25; cursor: default; }
    .modal-strip { display: flex; justify-content: center; gap: 6px; padding: 6px 12px; background: #222; flex-wrap: wrap; flex-shrink: 0; }
    .modal-strip img { width: 90px; height: 50px; object-fit: cover; border-radius: 4px; border: 2px solid transparent; cursor: pointer; flex-shrink: 0; }
    .modal-strip img.active { border-color: #0f766e; }
    .modal-bot { display: flex; justify-content: space-between; align-items: center; padding: 8px 14px; background: #222; color: #aaa; font-size: 12px; gap: 8px; flex-wrap: wrap; flex-shrink: 0; }
    .modal-bot .actions { display: flex; gap: 6px; }
    .btn-setthumb { background: #0f766e; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; margin-top: 0; width: auto; }
    .btn-setthumb:hover { background: #115e59; }
    .btn-revert-modal { background: #555; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; margin-top: 0; width: auto; }
    .btn-revert-modal:hover { background: #777; }
    .btn-close-modal { background: #333; color: #ccc; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; cursor: pointer; margin-top: 0; width: auto; }
    .btn-close-modal:hover { background: #444; }
    @media (max-width: 860px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid var(--line); } }
  </style>
</head>
<body>
  <header>
    <h1>CodeNova Retrieval UI</h1>
  </header>
  <main>
    <aside>
      <form id="search-form">
        <label for="track">Retrieval Track</label>
        <select id="track" name="track">
          <option value="textual_kis">Textual KIS</option>
          <option value="video_kis">Video KIS</option>
          <option value="vqa">VQA</option>
          <option value="trake">TRAKE</option>
        </select>

        <div id="vqa-settings" style="display: none;">
          <label for="vqa-backend">VQA Backend</label>
          <select id="vqa-backend" name="vqa_backend">
            <option value="internvl">InternVL (Local GPU)</option>
            <option value="gemini">Gemini (Cloud API)</option>
            <option value="ollama">Ollama (Local CPU/GPU)</option>
          </select>
        </div>

        <label for="query">Query</label>
        <textarea id="query" name="query">a person riding a motorbike</textarea>

        <label for="context">Scene / Context</label>
        <textarea id="context" name="context" placeholder="Optional shot sequence or scene description"></textarea>

        <label for="question">Question</label>
        <textarea id="question" name="question" placeholder="Use this for VQA or QA tracks"></textarea>

        <div class="row">
          <div>
            <label for="top-k">Top K</label>
            <input id="top-k" name="top_k" type="number" value="20" min="1" max="100">
          </div>
          <div>
            <label for="reranker-top-k">Reranker Top K</label>
            <input id="reranker-top-k" name="reranker_top_k" type="number" placeholder="Leave blank to disable" min="1" max="100">
          </div>
        </div>
        <button id="submit" type="submit">Search</button>
        <div id="sidebar-answer" style="display:none; margin-top: 14px; padding: 12px 14px; border: 1px solid var(--accent); border-radius: 8px; background: #f0fdf8;">
          <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);">Answer</div>
          <div id="sidebar-answer-text" style="margin-top: 4px; font-size: 15px; font-weight: 700; color: var(--accent-strong); line-height: 1.4;"></div>
        </div>
      </form>
      <p class="hint">VQA: 3-stage pipeline (Embedding search → Temporal → Agent). TRAKE: temporal search + event grouping. Textual/Video KIS: embedding search only.</p>
      <div id="status" class="status">Ready.</div>
    </aside>
    <section>
      <div id="answer-box"></div>
      <div id="events-box"></div>
      <div id="pipeline-box"></div>
      <div id="results" class="results"></div>
    </section>
  </main>
  <script>
    const form = document.getElementById("search-form");
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");
    const answerBox = document.getElementById("answer-box");
    const eventsEl = document.getElementById("events-box");
    const pipelineBox = document.getElementById("pipeline-box");
    const submitEl = document.getElementById("submit");
    const sidebarAnswer = document.getElementById("sidebar-answer");
    const sidebarAnswerText = document.getElementById("sidebar-answer-text");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      submitEl.disabled = true;
      statusEl.className = "status";
      statusEl.textContent = "Searching...";
      resultsEl.innerHTML = "";
      answerBox.innerHTML = "";
      eventsEl.innerHTML = "";
      pipelineBox.innerHTML = "";
      sidebarAnswer.style.display = "none";

  // ─── TRAKE event inputs ───────────────────────────────────────────────────────
  function eventCount() { return EVENTS_LIST.children.length; }
  function addSubInput(btn) {
    const wrapper = btn.closest(".event-wrapper");
    if (!wrapper) return;
    const container = wrapper.querySelector(".sub-inputs");
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:6px;margin:0 0 4px 28px;align-items:start;";
    row.innerHTML = `
      <textarea class="sub-detail-input" style="flex:1;min-height:36px;font-size:12px;" placeholder="Sub-detail..."></textarea>
      <button type="button" style="width:auto;padding:2px 8px;margin-top:0;font-size:12px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.parentElement.remove()" title="Remove">✕</button>`;
    container.appendChild(row);
  }
  function addEvent(value) {
    const idx = eventCount() + 1;
    const wrapper = document.createElement("div");
    wrapper.className = "event-wrapper";
    wrapper.style.cssText = "margin-bottom:8px;";
    wrapper.innerHTML = `
      <div style="display:flex;gap:6px;align-items:start;">
        <textarea class="event-input" style="flex:1;min-height:56px;" placeholder="Event ${idx} description">${esc(value||"")}</textarea>
        <button type="button" class="add-sub-btn" style="width:auto;padding:6px 10px;margin-top:0;font-size:16px;background:transparent;color:var(--accent);border-color:var(--accent);" onclick="addSubInput(this)" title="Add sub-detail">+</button>
        <button type="button" style="width:auto;padding:6px 10px;margin-top:0;font-size:14px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.closest('.event-wrapper').remove()" title="Remove">✕</button>
      </div>
      <div class="sub-inputs"></div>`;
    EVENTS_LIST.appendChild(wrapper);
  }
  eid("add-event-btn").addEventListener("click", () => addEvent(""));
  eid("window-slider").addEventListener("input", () => {
    eid("window-value").textContent = eid("window-slider").value;
  });
  function getEvents() {
    return Array.from(EVENTS_LIST.querySelectorAll(".event-wrapper"))
      .map(w => {
        const textarea = w.querySelector(".event-input");
        const text = textarea ? textarea.value.trim() : "";
        if (!text) return null;
        const subInputs = w.querySelectorAll(".sub-detail-input");
        const sub_details = Array.from(subInputs).map(s => s.value.trim()).filter(Boolean);
        return { text, sub_details };
      }).filter(Boolean);
  }

      if (track === "vqa") {
          payload.vqa_backend = form.vqa_backend.value;
      }
      if (form.reranker_top_k.value) {
        payload.reranker_top_k = Number(form.reranker_top_k.value);
      }

      let endpoint;
      if (track === "vqa") {
        endpoint = "/api/vqa-search";
      } else if (track === "trake") {
        endpoint = "/api/trake-search";
      } else {
        STATUS.innerHTML = `<strong>${data.results.length}</strong> results for <span class="pill">${track==="textual_kis"?"KIS Basic":"Video KIS"}</span>`;
        renderCards(data.results);
      }
    } catch(err) { STATUS.className="status warn"; STATUS.textContent=err.message; }
    finally { SUBMIT.disabled = false; }
  });

      const TIMEOUT_MS = 300_000;  // 5 min — models pre-warm in background but may still need time
      const MAX_RETRIES = 1;       // No auto-retry: models are pre-warmed at startup, not on first request
      let countdownTimer = null;

      function startCountdown(totalMs, attempt, maxAttempts) {
        let remaining = Math.ceil(totalMs / 1000);
        const attemptLabel = maxAttempts > 1 ? ` (attempt ${attempt}/${maxAttempts})` : "";
        statusEl.className = "status";
        statusEl.textContent = `Searching\u2026${attemptLabel} — timeout in ${remaining}s. First run loads models and may take several minutes.`;
        countdownTimer = setInterval(() => {
          remaining -= 1;
          if (remaining <= 0) {
            clearInterval(countdownTimer);
          } else {
            statusEl.textContent = `Searching\u2026${attemptLabel} — timeout in ${remaining}s.`;
          }
        }, 1000);
      }

      function stopCountdown() {
        if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null; }
      }

      async function fetchWithRetry(endpoint, payload, maxRetries) {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
          startCountdown(TIMEOUT_MS, attempt, maxRetries);

          try {
            const response = await fetch(endpoint, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "Bypass-Tunnel-Reminder": "true"
              },
              body: JSON.stringify(payload),
              signal: controller.signal
            });
            clearTimeout(timeoutId);
            stopCountdown();

            const text = await response.text();
            let data;
            try {
              data = JSON.parse(text);
            } catch (parseError) {
              // If it's not JSON (like "Bad Gateway" HTML), treat it as a server error
              if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${text.substring(0, 100).replace(/<[^>]*>?/gm, '')}`);
              }
              throw parseError;
            }

            // Treat 502/503/504 as retryable server errors (e.g. ngrok/colab gateway timeouts)
            if ([502, 503, 504].includes(response.status)) {
               throw new Error(`Gateway Error ${response.status}`);
            }

            return { response, data };
          } catch (err) {
            clearTimeout(timeoutId);
            stopCountdown();
            const isRetryable = err.name === "AbortError" ||
                                err.name === "TypeError" ||
                                err.message.includes("HTTP 502") ||
                                err.message.includes("HTTP 503") ||
                                err.message.includes("HTTP 504") ||
                                err.message.includes("Gateway Error");

            if (isRetryable && attempt < maxRetries) {
              statusEl.className = "status warn";
              statusEl.textContent = `Attempt ${attempt} failed (${err.message.substring(0, 40)}). Retrying automatically (${attempt}/${maxRetries})\u2026`;
              await new Promise(r => setTimeout(r, 2000)); // 2s pause before retry
              continue;
            }
            throw err; // final attempt or non-retryable error
          }
        }
      }

      try {
        const { response, data } = await fetchWithRetry(endpoint, payload, MAX_RETRIES);
        if (!response.ok || data.error) {
          throw new Error(data.error || "Search failed");
        }

        // Agent error: show dedicated error banner (separate from HTTP errors)
        if (track === "vqa" && data.agent_error) {
          statusEl.className = "status warn";
          statusEl.textContent = "Agent failed (see error below)";
          answerBox.innerHTML = `<div class="answer-box" style="border-color:#e55;background:#fff5f5">
            <div class="label" style="color:#c00">Agent Error</div>
            <div class="answer-text" style="color:#c00;font-size:13px;font-family:monospace">${escapeHtml(data.agent_error)}</div>
          </div>`;
          renderPipeline(data);
          renderResults(data.results || []);
          return;
        }

        if (track === "vqa" && data.answer) {
          statusEl.innerHTML = `<strong>Answer received</strong> via 3-stage pipeline <span class="pill">VQA</span>`;
          renderAnswer(data.answer);
          renderPipeline(data);
          renderResults(data.results || []);
          sidebarAnswer.style.display = "block";
          sidebarAnswerText.textContent = data.answer;
        } else if (track === "trake") {
          const eventCount = (data.events || []).length;
          statusEl.innerHTML = `<strong>${eventCount}</strong> event(s) found <span class="pill">TRAKE</span>`;
          renderTrakeEvents(data.events || []);
          renderPipeline(data);
          renderResults(data.results || []);
        } else {
          const trackLabel = track === "textual_kis" ? "Textual KIS" : "Video KIS";
          statusEl.innerHTML = `<strong>${data.results.length}</strong> results for <span class="pill">${trackLabel}</span>`;
          renderResults(data.results);
        }
      } catch (error) {
        stopCountdown();
        statusEl.className = "status warn";
        statusEl.textContent = (error.name === "AbortError" || error.name === "TypeError")
          ? `All ${MAX_RETRIES} attempts timed out. The server is still loading models — please wait a minute and try again.`
          : error.message;
      } finally {
        submitEl.disabled = false;
      }
    });

    function renderAnswer(answer) {
      answerBox.innerHTML = `
        <div class="answer-box">
          <div class="label">Answer</div>
          <div class="answer-text">${escapeHtml(answer)}</div>
        </div>
      </article>`).join("");
  }

    function renderPipeline(data) {
      const pipeline = data.pipeline || {};
      const hasAgent = pipeline.agent;
      const stages = hasAgent ? [
        { key: "embed_search", label: "Embedding Search", desc: `Top-${pipeline.embed_search?.top_k} frames retrieved` },
        { key: "temporal_search", label: "Temporal Search", desc: `${pipeline.temporal_search?.segments_found || 0} segments found` },
        { key: "gather_shot", label: "Shot Gather", desc: `${pipeline.gather_shot?.shots_count || 0} valid shots gathered` },
        { key: "shot_validation", label: "Shot Validation", desc: `Score: ${(pipeline.shot_validation?.validation_score || 0).toFixed(4)}` },
        { key: "agent", label: "Agent (Gemini)", desc: `Answer: ${(pipeline.agent?.answer || "N/A").substring(0, 100)}` },
      ] : [
        { key: "embed_search", label: "Embedding Search", desc: `Top-${pipeline.embed_search?.top_k} frames retrieved` },
        { key: "temporal_search", label: "Temporal Search", desc: `${pipeline.temporal_search?.segments_found || 0} segments found` },
        { key: "gather_shot", label: "Shot Gather", desc: `${pipeline.gather_shot?.shots_count || 0} valid shots gathered` },
      ];
      pipelineBox.innerHTML = `
        <button class="pipeline-toggle" onclick="togglePipeline()">Show Pipeline Details</button>
        <div id="pipeline-detail" class="pipeline-detail">
          ${stages.map((s, i) => `
            <div style="margin-bottom: 8px;">
              <strong>Stage ${i + 1}: ${escapeHtml(s.label)}</strong><br>
              ${escapeHtml(s.desc)}
            </div>
          `).join("")}
          ${hasAgent ? `<hr style="margin: 10px 0; border-color: var(--line);"><div><strong>Reasoning:</strong></div><code>${escapeHtml(data.reasoning || "N/A")}</code>` : ""}
        </div>
      `;
    }

    function renderTrakeEvents(events) {
      if (!events.length) return;
      const html = events.map((ev, i) => `
        <div style="margin-bottom: 16px; padding: 16px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel);">
          <div style="margin-bottom: 8px;"><strong>Event #${i + 1}</strong></div>
          <div style="font-size: 14px; margin-bottom: 4px;">Video: <strong>${escapeHtml(ev.video_name || ev.video_id || "")}</strong></div>
          <div style="font-size: 13px; color: var(--muted); margin-bottom: 4px;">
            Frames: ${ev.frame_count} · Time: ${formatTime(ev.start_timestamp)} - ${formatTime(ev.end_timestamp)}
          </div>
          <div style="font-size: 13px; color: var(--muted); margin-bottom: 12px;">Score: ${(ev.score || 0).toFixed(4)}</div>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            ${(ev.image_urls || []).map(url => `
              <img src="${escapeHtml(url)}" style="width: 100px; height: 56px; object-fit: cover; border-radius: 4px; border: 1px solid var(--line);" loading="lazy">
            `).join("")}
          </div>
        </div>
      `).join("");
      eventsEl.innerHTML = html;
    }

    function togglePipeline() {
      document.getElementById("pipeline-detail").classList.toggle("open");
    }

    function renderResults(results) {
      resultsEl.innerHTML = results.map((result, index) => `
        <article class="card">
          <img src="${escapeHtml(result.image_url || "")}" alt="Result frame ${index + 1}" loading="lazy">
          <div class="meta">
            <div><strong>#${index + 1}</strong> score ${Number(result.score).toFixed(4)}</div>
            <div>${formatTime(result.timestamp_sec)}</div>
            <div><strong>${escapeHtml(result.video_name || result.video_id || "")}</strong></div>
            <div>frame ${formatNumber(result.frame_index)} · shot ${escapeHtml(result.shot_id || "")}</div>
            <code>${escapeHtml(result.video_path || "")}</code>
            <code>${escapeHtml(result.frame_id || "")}</code>
          </div>
          <div class="event-grid" style="grid-template-columns:repeat(${cols},1fr)">${evHtml}</div>
        </div>`;
    }).join("");
  }

  function revertCard(videoId, chainIdx, eventIdx) {
    const key = trakeKey(videoId, chainIdx, eventIdx);
    const st = thumbState[key];
    if (!st) return;
    st.currentUrl = st.originalUrl;
    st.currentTimestamp = st.originalTimestamp;
    st.currentFrameId = st.originalFrameId;
    refreshCard(videoId, chainIdx, eventIdx);
    STATUS.innerHTML = `<strong>Reverted</strong> Event ${eventIdx+1} to original <span class="pill">ORIGINAL</span>`;
    // sync modal if it's open on this card
    if (modal.open && modal.videoId===videoId && modal.chainIdx===chainIdx && modal.eventIdx===eventIdx) {
      eid("btn-revert-modal").style.display = "none";
      eid("btn-setthumb").textContent = "Làm thumbnail";
    }
  }

  function refreshCard(videoId, chainIdx, eventIdx) {
    const key = trakeKey(videoId, chainIdx, eventIdx);
    const st = thumbState[key];
    if (!st) return;
    const cardId = "evcard-" + key.replaceAll("::","__");
    const card = eid(cardId);
    if (!card) return;
    const isCustom = st.currentUrl !== st.originalUrl;
    const img = card.querySelector("img");
    if (img) img.src = st.currentUrl;
    // Update the info text (timestamp changes when thumbnail changes)
    const textSpan = eid("evinfo-text-" + key.replaceAll("::","__"));
    if (textSpan) {
      textSpan.innerHTML = `<strong>Event ${eventIdx+1}</strong> &middot; rank ${st.rank} &middot; ${fmtTime(st.currentTimestamp)}`;
    }
    const info = card.querySelector(".ev-info");
    if (info) {
      const existing = info.querySelector(".pill");
      if (isCustom && !existing) {
        const span = document.createElement("span");
        span.className = "pill"; span.style.fontSize = "10px"; span.textContent = "CUSTOM";
        info.appendChild(span);
      } else if (!isCustom && existing) {
        existing.remove();
      }
    }
    if (isCustom) card.classList.add("has-custom");
    else card.classList.remove("has-custom");
  }

  // ─── Modal state ──────────────────────────────────────────────────────────────
  const modal = {
    open: false,
    shots: [],        // [{shot_id, frames:[{frame_id, frame_index, timestamp_sec, image_url}]}]
    shotIdx: 0,
    frameIdx: 0,
    videoId: "",
    chainIdx: null,   // null for non-trake cards
    eventIdx: null,   // null for non-trake cards
  };

  // ─── Open modal from card (reads live thumbState, not stale render values) ───
  function openModalFromCard(safeKey) {
    const key = safeKey.replaceAll('__','::');
    const st = thumbState[key];
    if (!st) return;
    openModal(st.currentUrl, st.videoId, st.currentFrameId, st.eventIdx, null, st.chainIdx);
  }

  // ─── Modal open ───────────────────────────────────────────────────────────────
  function openModal(imgSrc, videoId, frameId, eventIdx, _unused, chainIdx) {
    // Show modal immediately with the clicked image
    modal.open = true;
    modal.videoId = videoId;
    modal.chainIdx = chainIdx != null ? chainIdx : null;
    modal.eventIdx = eventIdx;
    modal.shots = [];
    modal.shotIdx = 0;
    modal.frameIdx = 0;

    const modalEl = eid("frame-modal");
    modalEl.style.display = "flex";

    const imgEl = eid("modal-img");
    imgEl.src = imgSrc;

    eid("modal-title").textContent = "Loading shots…";
    eid("modal-time").textContent = "";
    eid("modal-footer").textContent = "";
    eid("modal-strip").innerHTML = "";
    eid("modal-prev").disabled = true;
    eid("modal-next").disabled = true;
    // Only show thumbnail controls for TRAKE event cards (eventIdx is a number)
    const isTrake = eventIdx !== null && eventIdx !== undefined;
    eid("btn-setthumb").style.display = isTrake ? "inline-block" : "none";
    eid("btn-revert-modal").style.display = "none";

    if (!videoId) { eid("modal-title").textContent = "No video ID"; return; }

    fetch("/api/video-shots?video_id=" + encodeURIComponent(videoId))
      .then(r => r.json())
      .then(data => {
        const shots = data.shots || [];
        if (!shots.length) { eid("modal-title").textContent = "No shots found"; return; }
        modal.shots = shots;
        // Find the shot + frame matching frameId
        let foundSi = 0, foundFi = 0, found = false;
        for (let si = 0; si < shots.length && !found; si++) {
          const frames = shots[si].frames || [];
          for (let fi = 0; fi < frames.length; fi++) {
            if (frames[fi].frame_id === frameId) {
              foundSi = si; foundFi = fi; found = true; break;
            }
          }
        }
        modal.shotIdx = foundSi;
        modal.frameIdx = foundFi;
        renderModal();
      })
      .catch(() => {
        eid("modal-title").textContent = "Could not load shots";
      });
  }

  // ─── Modal render (called after shots loaded, or after navigation) ────────────
  function renderModal() {
    if (!modal.shots.length) return;
    const shot = modal.shots[modal.shotIdx];
    const frame = shot.frames[modal.frameIdx];

    // Set main image
    eid("modal-img").src = frame.image_url;

    // Header
    eid("modal-title").textContent =
      `Shot ${modal.shotIdx+1}/${modal.shots.length}  (${shot.shot_id})`;
    eid("modal-time").textContent = fmtTime(frame.timestamp_sec);

    // Footer
    eid("modal-footer").textContent =
      `Frame ${modal.frameIdx+1}/${shot.frames.length} · ${fmtTime(frame.timestamp_sec)} · idx ${frame.frame_index}`;

    // Strip
    eid("modal-strip").innerHTML = shot.frames.map((f, fi) =>
      `<img src="${esc(f.image_url)}" class="${fi===modal.frameIdx?"active":""}"
        onclick="selectStrip(event,${fi})" title="${fmtTime(f.timestamp_sec)}">`
    ).join("");

    // Nav buttons
    eid("modal-prev").disabled = modal.shotIdx <= 0;
    eid("modal-next").disabled = modal.shotIdx >= modal.shots.length - 1;

    // Thumbnail buttons (only for TRAKE)
    if (modal.eventIdx !== null && modal.eventIdx !== undefined) {
      const key = trakeKey(modal.videoId, modal.chainIdx, modal.eventIdx);
      const st = thumbState[key];
      const isCustom = st && st.currentUrl !== st.originalUrl;
      eid("btn-setthumb").textContent = isCustom ? "Cập nhật thumbnail" : "Làm thumbnail";
      eid("btn-revert-modal").style.display = isCustom ? "inline-block" : "none";
    }

    // Toggle VQA settings visibility
    form.track.addEventListener('change', (e) => {
      document.getElementById('vqa-settings').style.display = e.target.value === 'vqa' ? 'block' : 'none';
    });
    // Trigger initially
    if (form.track.value === 'vqa') {
        document.getElementById('vqa-settings').style.display = 'block';
    }
  </script>
</body>
</html>
"""
