"""Small stdlib web UI for query and result image inspection."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html as html_lib
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import json
from core.logging import get_logger
import mimetypes

from config.settings import Experiment
from core.errors import RetrievalError
from core.types import SearchResult
from indexing.readiness import read_readiness
from indexing.validation import verify_artifact_fingerprints, verify_frame_files
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
)
from ui.views.page import INDEX_HTML

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class WarmupComponentHealth:
    component: str
    status: str
    error: str | None = None


@dataclass(frozen=True)
class WarmupReport:
    experiment: str
    status: str
    components: tuple[WarmupComponentHealth, ...]
    ui_reranker: object | None


class _FailOpenReranker:
    """Disable an optional UI reranker after its first runtime failure."""

    def __init__(self, reranker) -> None:
        self._reranker = reranker
        self._available = True

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not self._available:
            return results
        try:
            return self._reranker.rerank(query=query, results=results)
        except Exception as exc:
            self._available = False
            LOGGER.exception(
                "event=RERANKER_DEGRADED component=ui-reranker error=%s; "
                "returning pre-rerank results",
                exc,
            )
            return results


def _warmup_models(reranker, experiment: Experiment, retriever=None) -> WarmupReport:
    """Pre-load heavy models before the server starts accepting requests.

    The lazy-loaded models (BLIP-2 reranker, the embedders) can take minutes to
    download and place on the GPU. Loading them up front keeps the first query
    from paying that cost, and — because this runs to completion before the
    listener starts — no two threads can race to initialize the same model.
    """
    LOGGER.info("event=WARMUP_STARTED experiment=%s", experiment.name)
    health: list[WarmupComponentHealth] = []
    failed_embedders: list[str] = []

    # Every configured embedder is mandatory for the selected experiment, but
    # each one is checked independently so one failure cannot hide later ones.
    if retriever is not None:
        for model_name, embedder in retriever.embedders.items():
            component = f"embedder:{model_name}"
            try:
                embedder.embed_text("warmup query")
            except Exception as exc:
                failed_embedders.append(model_name)
                health.append(WarmupComponentHealth(component, "FAILED", str(exc)))
                LOGGER.exception(
                    "event=WARMUP_COMPONENT_FAILED component=%s error=%s", component, exc
                )
            else:
                health.append(WarmupComponentHealth(component, "READY"))
                LOGGER.info("event=WARMUP_COMPONENT_READY component=%s", component)

    def warm_optional(component: str, candidate):
        if candidate is None:
            return None
        try:
            loader = getattr(candidate, "_load", None)
            if callable(loader):
                loader()
        except Exception as exc:
            health.append(WarmupComponentHealth(component, "FAILED", str(exc)))
            LOGGER.exception(
                "event=WARMUP_COMPONENT_FAILED component=%s error=%s; disabled=true",
                component,
                exc,
            )
            return None
        health.append(WarmupComponentHealth(component, "READY"))
        LOGGER.info("event=WARMUP_COMPONENT_READY component=%s", component)
        return candidate

    healthy_ui_reranker = warm_optional("reranker:ui", reranker)
    if retriever is not None and retriever.reranker is not None:
        healthy_internal = warm_optional("reranker:retriever", retriever.reranker)
        if healthy_internal is None:
            retriever.reranker = None

    if failed_embedders:
        LOGGER.error(
            "event=WARMUP_FAILED experiment=%s mandatory_embedders=%s",
            experiment.name,
            failed_embedders,
        )
        raise RetrievalError(
            "Mandatory experiment embedder warmup failed: " + ", ".join(failed_embedders)
        )

    status = "DEGRADED" if any(item.status == "FAILED" for item in health) else "READY"
    LOGGER.info("event=WARMUP_COMPLETED experiment=%s status=%s", experiment.name, status)
    return WarmupReport(
        experiment.name,
        status,
        tuple(health),
        _FailOpenReranker(healthy_ui_reranker) if healthy_ui_reranker is not None else None,
    )


def _validate_experiment_for_serving(experiment: Experiment) -> dict[str, object]:
    """Require a fresh offline readiness report without mutating indexing artifacts."""
    readiness = read_readiness(experiment)
    status = str(readiness.get("status", "INVALID"))
    # DEGRADED được chấp nhận: chỉ có WARNING (vd thiếu video gốc, không cần
    # cho search/keyframe) — chỉ chặn khi INVALID (còn ERROR thật sự).
    if status not in ("READY", "DEGRADED"):
        raise RuntimeError(
            f"Experiment {experiment.name!r} is not ready: status={status}. "
            "Run validate-index and resolve the reported issues first."
        )
    if readiness.get("config_hash") != experiment.config.config_hash():
        raise RuntimeError(
            f"Experiment {experiment.name!r} readiness config does not match persisted config. "
            "Run validate-index again."
        )
    stale = verify_artifact_fingerprints(readiness)
    if stale:
        raise RuntimeError(
            f"Experiment {experiment.name!r} readiness is stale for artifacts: {stale}. "
            "Run validate-index again."
        )
    frame_errors = verify_frame_files(experiment)
    if frame_errors:
        sample = "; ".join(f"{issue['frame_id']}:{issue['reason']}" for issue in frame_errors[:20])
        raise RuntimeError(
            f"Experiment has {len(frame_errors)} invalid frame artifact(s): {sample}"
        )
    return readiness


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

    readiness = _validate_experiment_for_serving(experiment)
    LOGGER.info(
        "[readiness] Validated experiment=%s generated_at=%s",
        experiment.name,
        readiness.get("generated_at"),
    )
    # Ensure manifests exist so UI never has broken frame_path lookups
    ensure_manifests(experiment)

    # frame_path values in manifests are relative to the working directory the
    # pipeline was run from, not to experiment.run_dir. Resolve against cwd.
    set_project_root(Path.cwd())

    from retrieval.vqa import _get_retriever

    retriever = _get_retriever(experiment)
    # Warm before constructing the handler so a failed optional reranker is not
    # captured by request handlers and mandatory failures prevent socket bind.
    warmup = _warmup_models(reranker, experiment, retriever)
    handler = build_handler(
        experiment=experiment,
        retriever=retriever,
        default_top_k=default_top_k,
        reranker=warmup.ui_reranker,
        reranker_top_k=reranker_top_k,
    )
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
                html = _render_index_html(experiment, default_top_k)
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
                    res = handle_vqa_search(
                        payload, experiment, default_top_k, reranker, reranker_top_k
                    )
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
            from core.paths import resolve_experiment_frame_path

            raw_path = unquote(raw_path)
            if Path(raw_path).suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                self.send_error(HTTPStatus.NOT_FOUND, "Frame not found")
                return
            resolution = resolve_experiment_frame_path(experiment, raw_path)
            if not resolution.valid or resolution.resolved_path is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Frame not found")
                return
            frame_path = resolution.resolved_path

            content_type = mimetypes.guess_type(frame_path.name)[0] or "application/octet-stream"
            content = frame_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return RetrievalUiHandler


def _render_index_html(experiment: Experiment, default_top_k: int) -> str:
    """Render the active persisted experiment identity into the UI shell."""
    return (
        INDEX_HTML.replace('value="20"', f'value="{default_top_k}"')
        .replace("__ACTIVE_EXPERIMENT__", html_lib.escape(experiment.name))
        .replace(
            "__ACTIVE_MODELS__",
            html_lib.escape(", ".join(experiment.config.embedding_models)),
        )
    )
