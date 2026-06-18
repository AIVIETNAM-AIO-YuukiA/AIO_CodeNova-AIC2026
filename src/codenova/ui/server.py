"""Small stdlib web UI for query and result image inspection."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
import json
from codenova.core.logging import get_logger
import mimetypes

from codenova.config.settings import Experiment
from codenova.core.types import SearchResult
from codenova.retrieval import build_retriever
from codenova.retrieval.tracks import SUPPORTED_TRACKS, TrackQuery, build_retrieval_text

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
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/search":
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
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #1c1f24;
      --muted: #667085;
      --line: #d9dde3;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --warn: #a16207;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 20px; line-height: 1.2; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      min-height: calc(100vh - 61px);
    }
    aside {
      padding: 18px;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
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
      grid-template-columns: 1fr 112px;
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
    .hint, .status {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .status strong { color: var(--text); }
    .status.warn { color: var(--warn); }
    .results {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 14px;
    }
    .card {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .card img {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      display: block;
      background: #e5e7eb;
    }
    .meta {
      padding: 10px 11px 12px;
      font-size: 13px;
      line-height: 1.45;
    }
    .meta code {
      display: block;
      overflow-wrap: anywhere;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #e6f5f3;
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 700;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
    }
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
          <option value="vqa">VQA</option>
          <option value="qa">Question Answering</option>
          <option value="visual_kis">Visual KIS</option>
        </select>

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
          <button id="submit" type="submit">Search</button>
        </div>
      </form>
      <p class="hint">Current backend routes all tracks through CLIP text-to-frame search. VQA fields are preserved so the backend can later add answer generation and evidence ranking without changing the UI contract.</p>
      <div id="status" class="status">Ready.</div>
    </aside>
    <section>
      <div id="results" class="results"></div>
    </section>
  </main>
  <script>
    const form = document.getElementById("search-form");
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");
    const submitEl = document.getElementById("submit");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      submitEl.disabled = true;
      statusEl.className = "status";
      statusEl.textContent = "Searching...";
      resultsEl.innerHTML = "";

      const payload = {
        track: form.track.value,
        query: form.query.value,
        context: form.context.value,
        question: form.question.value,
        top_k: Number(form.top_k.value || 20)
      };

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || data.error) {
          throw new Error(data.error || "Search failed");
        }
        statusEl.innerHTML = `<strong>${data.results.length}</strong> results for <span class="pill">${data.track_label}</span>`;
        renderResults(data.results);
      } catch (error) {
        statusEl.className = "status warn";
        statusEl.textContent = error.message;
      } finally {
        submitEl.disabled = false;
      }
    });

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
        </article>
      `).join("");
    }

    function formatTime(value) {
      if (value === null || value === undefined) return "";
      const seconds = Number(value);
      const minutes = Math.floor(seconds / 60);
      const rest = Math.round(seconds % 60);
      return `${minutes}:${String(rest).padStart(2, "0")} (${seconds.toFixed(2)}s)`;
    }

    function formatNumber(value) {
      if (value === null || value === undefined) return "unknown";
      return Number(value).toLocaleString();
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""
