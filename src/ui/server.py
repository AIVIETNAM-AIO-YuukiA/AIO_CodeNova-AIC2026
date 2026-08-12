"""Small stdlib web UI for query and result image inspection."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import json
from core.logging import get_logger
import mimetypes

from config.settings import Experiment
from ui.api import (
    ensure_manifests,
    handle_agent_chat,
    handle_compute_sub_score,
    handle_default_search,
    handle_intelligent_search,
    handle_kis_detail_2stage,
    handle_text_search,
    handle_trake_or_enhanced_search,
    handle_video_shots,
    handle_vqa_search,
    warmup_models,
)
from ui.views.page import INDEX_HTML

LOGGER = get_logger(__name__)


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

    # Ensure manifests exist so UI never has broken frame_path lookups
    ensure_manifests(experiment)

    # frame_path values in manifests are relative to the working directory the
    # pipeline was run from, not to experiment.run_dir. Resolve against cwd.
    set_project_root(Path.cwd())

    from retrieval.vqa import _get_retriever

    retriever = _get_retriever(experiment)
    handler = build_handler(
        experiment=experiment,
        retriever=retriever,
        default_top_k=default_top_k,
        reranker=reranker,
        reranker_top_k=reranker_top_k,
    )
    warmup_models(reranker, experiment, retriever)

    server = ThreadingHTTPServer((host, port), handler)
    LOGGER.info("Serving retrieval UI at http://%s:%s", host, port)
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
                res, status = handle_video_shots(parse_qs(parsed.query), experiment)
                self._send_json(res, status=status)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)

            if parsed.path in ("/api/trake-search", "/api/enhanced-temporal-search"):
                try:
                    payload = self._read_json()
                    res = handle_trake_or_enhanced_search(
                        parsed.path, payload, experiment, default_top_k, reranker, reranker_top_k
                    )
                    self._send_json(res)
                except Exception as exc:
                    LOGGER.exception("Temporal search failed (%s)", parsed.path)
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/vqa-search":
                try:
                    payload = self._read_json()
                    res = handle_vqa_search(payload, experiment, default_top_k, reranker, reranker_top_k)
                    self._send_json(res)
                except Exception as exc:
                    LOGGER.exception("VQA search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/agent/chat":
                try:
                    payload = self._read_json()
                    res = handle_agent_chat(payload, experiment)
                    self._send_json(res)
                except Exception as exc:
                    LOGGER.exception("Agent chat failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path in ("/api/asr-search", "/api/ocr-search"):
                try:
                    payload = self._read_json()
                    res = handle_text_search(parsed.path, payload, experiment, default_top_k)
                    self._send_json(res)
                except Exception as exc:
                    LOGGER.exception("Text search failed (%s)", parsed.path)
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/intelligent-search":
                try:
                    payload = self._read_json()
                    res = handle_intelligent_search(payload, experiment, default_top_k)
                    self._send_json(res)
                except Exception as exc:
                    LOGGER.exception("Intelligent search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/kis-detail-2stage":
                try:
                    payload = self._read_json()
                    res = handle_kis_detail_2stage(payload, experiment)
                    self._send_json(res)
                except Exception as exc:
                    LOGGER.exception("KIS Detail 2-Stage search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/compute-sub-score":
                try:
                    payload = self._read_json()
                    res, status = handle_compute_sub_score(payload, experiment, retriever)
                    self._send_json(res, status=status)
                except Exception as exc:
                    LOGGER.exception("compute-sub-score failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path != "/api/search":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            try:
                payload = self._read_json()
                res = handle_default_search(payload, experiment, retriever, default_top_k)
                self._send_json(res)
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
            from core.paths import resolve_frame_path

            raw_path = unquote(raw_path)
            if Path(raw_path).suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                self.send_error(HTTPStatus.NOT_FOUND, "Frame not found")
                return
            frame_path = resolve_frame_path(raw_path)
            if not frame_path.is_file():
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
