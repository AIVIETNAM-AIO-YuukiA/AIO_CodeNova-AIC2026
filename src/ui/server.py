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
from retrieval import build_retriever
from retrieval.tracks import SUPPORTED_TRACKS, TrackQuery, build_retrieval_text

LOGGER = get_logger(__name__)


def serve_ui(
    experiment: Experiment,
    host: str = "127.0.0.1",
    port: int = 7860,
    default_top_k: int = 20,
) -> None:
    """Serve the local retrieval UI until interrupted."""
    retriever = build_retriever(experiment)
    handler = build_handler(experiment=experiment, retriever=retriever, default_top_k=default_top_k)
    server = ThreadingHTTPServer((host, port), handler)
    LOGGER.info("Serving retrieval UI at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping retrieval UI")
    finally:
        server.server_close()


def build_handler(experiment: Experiment, retriever, default_top_k: int):
    """Create a request handler bound to one experiment and its retriever."""

    class RetrievalUiHandler(BaseHTTPRequestHandler):
        server_version = "CodeNovaRetrievalUI/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
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

            # TRAKE: multi-event → per-event search → intersection → scoring
            if parsed.path == "/api/trake-search":
                try:
                    payload = self._read_json()
                    events_raw = payload.get("events")
                    if not isinstance(events_raw, list) or len(events_raw) < 2:
                        raise ValueError("At least 2 events are required.")
                    events: list[str] = [str(e).strip() for e in events_raw if str(e).strip()]
                    # Fixed to 300 frames for TRAKE (hard-coded, user cannot override)
                    top_k = 300
                    result = trake_search(
                        experiment=experiment,
                        events=events,
                        top_k=top_k,
                    )
                    for video in result.get("videos", []):
                        for ev in video.get("events", []):
                            if ev.get("frame_path"):
                                ev["image_url"] = f"/frame?path={quote(ev['frame_path'])}"
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("TRAKE search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # VQA track uses full pipeline (CLIP → Temporal → Agent)
            if parsed.path == "/api/vqa-search":
                try:
                    payload = self._read_json()
                    top_k = int(payload.get("top_k") or default_top_k)
                    result = vqa_search(
                        experiment=experiment,
                        query=str(payload.get("query", "")),
                        question=str(payload.get("question", "")),
                        context=str(payload.get("context", "")),
                        top_k=top_k,
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
                retrieval_text = build_retrieval_text(request)
                results = retriever.search(query=retrieval_text, top_k=top_k)
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
            frame_path = Path(unquote(raw_path)).resolve()
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
    label { display: block; margin: 14px 0 6px; color: var(--muted); font-size: 13px; font-weight: 650; }
    select, input, textarea, button { width: 100%; border: 1px solid var(--line); border-radius: 6px; font: inherit; }
    select, input, textarea { padding: 10px 11px; background: #fff; color: var(--text); }
    textarea { min-height: 96px; resize: vertical; line-height: 1.45; }
    .row { display: grid; grid-template-columns: 1fr 112px; gap: 10px; align-items: end; }
    button { margin-top: 16px; padding: 11px 14px; border-color: var(--accent); background: var(--accent); color: #fff; font-weight: 750; cursor: pointer; }
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
<header><h1>CodeNova Retrieval UI</h1></header>
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
      <label for="query">Query</label>
      <textarea id="query" name="query">a person riding a motorbike</textarea>
      <label for="context">Scene / Context</label>
      <textarea id="context" name="context" placeholder="Optional shot sequence or scene description"></textarea>
      <label for="question">Question</label>
      <textarea id="question" name="question" placeholder="Use this for VQA or QA tracks"></textarea>
      <div id="events-section" style="display:none;">
        <label>Events <span style="font-weight:400;color:var(--muted);">(each event is searched independently)</span></label>
        <div id="events-list"></div>
        <button type="button" id="add-event-btn" style="width:auto;padding:6px 14px;margin-top:6px;font-size:13px;">+ Add Event</button>
        <div class="hint" style="margin-top:8px;font-size:12px;">TRAKE uses 100 frames per event (hard-coded)</div>
      </div>
      <div class="row">
        <div>
          <label for="top-k">Top K</label>
          <input id="top-k" name="top_k" type="number" value="20" min="1" max="100">
        </div>
        <button id="submit" type="submit">Search</button>
      </div>
      <div id="sidebar-answer" style="display:none;margin-top:14px;padding:12px 14px;border:1px solid var(--accent);border-radius:8px;background:#f0fdf8;">
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);">Answer</div>
        <div id="sidebar-answer-text" style="margin-top:4px;font-size:15px;font-weight:700;color:var(--accent-strong);line-height:1.4;"></div>
      </div>
    </form>
    <p class="hint">VQA: 3-stage pipeline (CLIP → Temporal → Agent). TRAKE: multi-event temporal search. Textual/Video KIS: CLIP only.</p>
    <div id="status" class="status">Ready.</div>
  </aside>
  <section>
    <div id="answer-box"></div>
    <div id="pipeline-box"></div>
    <div id="results" class="results"></div>
    <!-- Modal -->
    <div id="frame-modal">
      <div class="modal-box" id="modal-box">
        <div class="modal-top">
          <span id="modal-title">Loading...</span>
          <span id="modal-time" class="time-badge"></span>
          <button class="close-x" id="modal-close-x">&times;</button>
        </div>
        <div class="modal-mid">
          <button class="modal-nav" id="modal-prev" title="Shot trước (←)">&#9664;</button>
          <div class="img-area">
            <img id="modal-img" src="" alt="Frame preview">
          </div>
          <button class="modal-nav" id="modal-next" title="Shot tiếp (→)">&#9654;</button>
        </div>
        <div class="modal-strip" id="modal-strip"></div>
        <div class="modal-bot">
          <span id="modal-footer"></span>
          <div class="actions">
            <button class="btn-setthumb" id="btn-setthumb">Làm thumbnail</button>
            <button class="btn-revert-modal" id="btn-revert-modal" style="display:none">Revert</button>
            <button class="btn-close-modal" id="btn-close-modal">Đóng (Esc)</button>
          </div>
        </div>
      </div>
    </div>
  </section>
</main>
<script>
  // ─── Helpers ─────────────────────────────────────────────────────────────────
  const eid = id => document.getElementById(id);
  function esc(v) {
    return String(v).replaceAll("&","&amp;").replaceAll("<","&lt;")
      .replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;");
  }
  function fmtTime(v) {
    if (v == null) return "";
    const s = Number(v), m = Math.floor(s / 60), r = Math.round(s % 60);
    return `${m}:${String(r).padStart(2,"0")} (${s.toFixed(2)}s)`;
  }
  function fmtNum(v) { return v == null ? "unknown" : Number(v).toLocaleString(); }

  // ─── DOM refs ────────────────────────────────────────────────────────────────
  const FORM          = eid("search-form");
  const STATUS        = eid("status");
  const RESULTS       = eid("results");
  const ANSWER_BOX    = eid("answer-box");
  const PIPELINE_BOX  = eid("pipeline-box");
  const SUBMIT        = eid("submit");
  const SIDEBAR_ANS   = eid("sidebar-answer");
  const SIDEBAR_TEXT  = eid("sidebar-answer-text");
  const EVENTS_SEC    = eid("events-section");
  const EVENTS_LIST   = eid("events-list");

  // ─── TRAKE event inputs ───────────────────────────────────────────────────────
  function eventCount() { return EVENTS_LIST.children.length; }
  function addEvent(value) {
    const idx = eventCount() + 1;
    const div = document.createElement("div");
    div.style.cssText = "display:flex;gap:6px;margin-bottom:6px;align-items:start;";
    div.innerHTML = `
      <textarea class="event-input" style="flex:1;min-height:56px;" placeholder="Event ${idx} description">${esc(value||"")}</textarea>
      <button type="button" style="width:auto;padding:6px 10px;margin-top:0;font-size:14px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.parentElement.remove()" title="Remove">✕</button>`;
    EVENTS_LIST.appendChild(div);
  }
  eid("add-event-btn").addEventListener("click", () => addEvent(""));
  function getEvents() {
    return Array.from(EVENTS_LIST.querySelectorAll(".event-input"))
      .map(i => i.value.trim()).filter(Boolean);
  }

  // ─── Track selector ───────────────────────────────────────────────────────────
  FORM.track.addEventListener("change", () => {
    const t = FORM.track.value, isT = t==="trake", isV = t==="vqa";
    eid("query").style.display = isT ? "none" : "";
    FORM.querySelector("label[for=query]").style.display = isT ? "none" : "";
    eid("context").style.display = isT||isV ? "" : "none";
    FORM.querySelector("label[for=context]").style.display = isT||isV ? "" : "none";
    eid("question").style.display = isV ? "" : "none";
    FORM.querySelector("label[for=question]").style.display = isV ? "" : "none";
    eid("top-k").style.display = isT ? "none" : "";
    FORM.querySelector("label[for=top-k]").style.display = isT ? "none" : "";
    EVENTS_SEC.style.display = isT ? "" : "none";
    if (isT && eventCount()===0) { addEvent("a person riding a motorbike"); addEvent("a person falling off"); }
  });
  FORM.track.dispatchEvent(new Event("change"));

  // ─── Search submit ────────────────────────────────────────────────────────────
  FORM.addEventListener("submit", async e => {
    e.preventDefault();
    SUBMIT.disabled = true;
    STATUS.className = "status"; STATUS.textContent = "Searching...";
    RESULTS.innerHTML = ""; ANSWER_BOX.innerHTML = ""; PIPELINE_BOX.innerHTML = "";
    SIDEBAR_ANS.style.display = "none";
    const track = FORM.track.value;
    let endpoint, payload;
    if (track === "trake") {
      const events = getEvents();
      if (events.length < 2) { STATUS.className="status warn"; STATUS.textContent="Need at least 2 events."; SUBMIT.disabled=false; return; }
      endpoint = "/api/trake-search"; payload = { events, top_k: 50 };
    } else if (track === "vqa") {
      endpoint = "/api/vqa-search";
      payload = { query: FORM.query.value, context: FORM.context.value, question: FORM.question.value, top_k: Number(FORM.top_k.value||20) };
    } else {
      endpoint = "/api/search";
      payload = { track, query: FORM.query.value, context: FORM.context.value, question: FORM.question.value, top_k: Number(FORM.top_k.value||20) };
    }
    try {
      const res = await fetch(endpoint, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Search failed");
      if (track === "trake") {
        const chains = data.videos || [];
        const uniqueVideos = new Set(chains.map(v => v.video_id)).size;
        STATUS.innerHTML = `<strong>${chains.length}</strong> chain(s) from <strong>${uniqueVideos}</strong> video(s) match all events <span class="pill">TRAKE</span>`;
        renderTrake(data);
      } else if (track === "vqa" && data.answer) {
        STATUS.innerHTML = `<strong>Answer received</strong> via 3-stage pipeline <span class="pill">VQA</span>`;
        ANSWER_BOX.innerHTML = `<div class="answer-box"><div class="label">Answer</div><div class="answer-text">${esc(data.answer)}</div></div>`;
        renderPipeline(data);
        renderCards(data.results||[]);
        SIDEBAR_ANS.style.display = "block"; SIDEBAR_TEXT.textContent = data.answer;
      } else {
        STATUS.innerHTML = `<strong>${data.results.length}</strong> results for <span class="pill">${track==="textual_kis"?"Textual KIS":"Video KIS"}</span>`;
        renderCards(data.results);
      }
    } catch(err) { STATUS.className="status warn"; STATUS.textContent=err.message; }
    finally { SUBMIT.disabled = false; }
  });

  function renderPipeline(data) {
    const p = data.pipeline||{}, ha = p.agent;
    const stages = ha ? [
      {l:"CLIP Search",d:`Top-${p.clip_search?.top_k} frames`},
      {l:"Temporal Search",d:`${JSON.stringify(p.temporal_search?.top_k_results||"N/A")}`},
      {l:"Shot Gather",d:`${p.gather_shot?.frame_count||0} frames in shot`},
      {l:"Shot Validation",d:`Score: ${(p.shot_validation?.validation_score||0).toFixed(4)}`},
      {l:"Agent (Gemini)",d:`${(p.agent?.answer||"N/A").substring(0,100)}`},
    ] : [
      {l:"CLIP Search",d:`Top-${p.clip_search?.top_k} frames`},
      {l:"Temporal Search",d:`${JSON.stringify(p.temporal_search?.top_k_results||"N/A")}`},
      {l:"Shot Gather",d:`${p.gather_shot?.frame_count||0} frames in shot`},
    ];
    PIPELINE_BOX.innerHTML = `<button class="pipeline-toggle" onclick="eid('pipeline-detail').classList.toggle('open')">Show Pipeline Details</button>
      <div id="pipeline-detail" class="pipeline-detail">
        ${stages.map((s,i)=>`<div style="margin-bottom:8px"><strong>Stage ${i+1}: ${esc(s.l)}</strong><br>${esc(s.d)}</div>`).join("")}
        ${ha?`<hr style="margin:10px 0;border-color:var(--line)"><strong>Reasoning:</strong><code>${esc(data.reasoning||"N/A")}</code>`:""}
      </div>`;
  }

  function renderCards(results) {
    RESULTS.innerHTML = results.map((r,i) => `
      <article class="card">
        <img src="${esc(r.image_url||"")}" alt="Result ${i+1}" loading="lazy"
          onclick="openModal('${esc(r.image_url||"")}','${esc(r.video_id||"")}','${esc(r.frame_id||"")}',null,null)">
        <div class="meta">
          <div><strong>#${i+1}</strong> score ${Number(r.score).toFixed(4)}</div>
          <div>${fmtTime(r.timestamp_sec)}</div>
          <div><strong>${esc(r.video_name||r.video_id||"")}</strong></div>
          <div>frame ${fmtNum(r.frame_index)} · shot ${esc(r.shot_id||"")}</div>
          <code>${esc(r.video_path||"")}</code>
          <code>${esc(r.frame_id||"")}</code>
        </div>
      </article>`).join("");
  }

  // ─── TRAKE render ─────────────────────────────────────────────────────────────
  // Per-card thumbnail state: key = "videoId::chainIdx::eventIdx"
  // Value: { originalUrl, currentUrl, originalFrameId, chainIdx, eventIdx }
  const thumbState = {};

  function trakeKey(videoId, chainIdx, eventIdx) { return videoId + "::" + chainIdx + "::" + eventIdx; }

  function renderTrake(data) {
    const videos = data.videos || [];
    if (!videos.length) {
      RESULTS.innerHTML = `<div class="hint" style="padding:20px;text-align:center;">No video found matching all events.</div>`;
      return;
    }
    RESULTS.innerHTML = videos.map((video, vi) => {
      const cols = Math.min(video.events.length, 5);
      const evHtml = (video.events||[]).map((ev, ei) => {
        const key = trakeKey(video.video_id, vi, ei);
        if (!thumbState[key]) {
          thumbState[key] = {
            originalUrl: ev.image_url||"", originalFrameId: ev.frame_id||"",
            originalTimestamp: ev.timestamp_sec,
            currentUrl: ev.image_url||"", currentTimestamp: ev.timestamp_sec,
            currentFrameId: ev.frame_id||"",
            rank: ev.rank,
            videoId: video.video_id,
            chainIdx: vi,
            eventIdx: ei,
          };
        }
        const st = thumbState[key];
        const url = st.currentUrl;
        const isCustom = url !== st.originalUrl;
        const safeKey = key.replaceAll("::","__");
        return `
          <div class="ev-card${isCustom?" has-custom":""}" id="evcard-${safeKey}">
            <img src="${esc(url)}" alt="Event ${ei+1}" loading="lazy"
              id="evimg-${safeKey}"
              onclick="openModalFromCard('${safeKey}')">
            <button class="revert-badge" onclick="revertCard('${esc(video.video_id)}',${vi},${ei})">&#x21a9; Revert</button>
            <div class="ev-info">
              <span id="evinfo-text-${esc(key.replaceAll("::","__"))}"><strong>Event ${ei+1}</strong> &middot; rank ${st.rank} &middot; ${fmtTime(st.currentTimestamp)}</span>
              ${isCustom?`<span class="pill" style="font-size:10px">CUSTOM</span>`:""}
            </div>
          </div>`;
      }).join("");
      return `
        <div class="video-block">
          <div class="video-block-header">
            <strong style="font-size:15px">#${vi+1} ${esc(video.video_name||video.video_id)}</strong>
            <span class="pill">Score: ${video.score} ${video.temporal_order_valid?"✓ temporal":"✗ temporal"}</span>
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
  }

  // ─── Strip click ─────────────────────────────────────────────────────────────
  function selectStrip(e, fi) {
    e.stopPropagation();
    modal.frameIdx = fi;
    renderModal();
  }

  // ─── Shot navigation ─────────────────────────────────────────────────────────
  eid("modal-prev").addEventListener("click", e => {
    e.stopPropagation();
    if (modal.shotIdx > 0) { modal.shotIdx--; modal.frameIdx = 0; renderModal(); }
  });
  eid("modal-next").addEventListener("click", e => {
    e.stopPropagation();
    if (modal.shotIdx < modal.shots.length-1) { modal.shotIdx++; modal.frameIdx = 0; renderModal(); }
  });

  // ─── Set thumbnail ───────────────────────────────────────────────────────────
  eid("btn-setthumb").addEventListener("click", () => {
    if (!modal.shots.length || modal.eventIdx === null) return;
    const frame = modal.shots[modal.shotIdx].frames[modal.frameIdx];
    const key = trakeKey(modal.videoId, modal.chainIdx, modal.eventIdx);
    if (!thumbState[key]) return;
    thumbState[key].currentUrl = frame.image_url;
    thumbState[key].currentTimestamp = frame.timestamp_sec;
    thumbState[key].currentFrameId = frame.frame_id;
    refreshCard(modal.videoId, modal.chainIdx, modal.eventIdx);
    renderModal(); // update button text + revert visibility
    STATUS.innerHTML = `<strong>Thumbnail updated</strong> — Event ${modal.eventIdx+1} <span class="pill">CUSTOM</span>`;
  });

  // ─── Revert from modal ───────────────────────────────────────────────────────
  eid("btn-revert-modal").addEventListener("click", () => {
    if (modal.eventIdx === null) return;
    revertCard(modal.videoId, modal.chainIdx, modal.eventIdx);
    renderModal();
  });

  // ─── Close modal ─────────────────────────────────────────────────────────────
  function closeModal() {
    eid("frame-modal").style.display = "none";
    eid("modal-img").src = "";
    modal.open = false;
    modal.shots = [];
  }
  eid("modal-close-x").addEventListener("click", closeModal);
  eid("btn-close-modal").addEventListener("click", closeModal);
  eid("frame-modal").addEventListener("click", e => {
    if (e.target === eid("frame-modal")) closeModal();
  });
  eid("modal-box").addEventListener("click", e => e.stopPropagation());

  // ─── Keyboard shortcuts ───────────────────────────────────────────────────────
  document.addEventListener("keydown", e => {
    if (!modal.open) return;
    if (e.key === "Escape") { closeModal(); return; }
    if (e.key === "ArrowLeft" && modal.shotIdx > 0) { modal.shotIdx--; modal.frameIdx=0; renderModal(); }
    if (e.key === "ArrowRight" && modal.shotIdx < modal.shots.length-1) { modal.shotIdx++; modal.frameIdx=0; renderModal(); }
    if (e.key === "ArrowUp" && modal.frameIdx > 0) { modal.frameIdx--; renderModal(); }
    if (e.key === "ArrowDown") {
      const maxFi = (modal.shots[modal.shotIdx]?.frames.length||1)-1;
      if (modal.frameIdx < maxFi) { modal.frameIdx++; renderModal(); }
    }
  });
</script>
</body>
</html>
"""
