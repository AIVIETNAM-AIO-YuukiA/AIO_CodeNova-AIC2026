"""Small stdlib web UI for query and result image inspection."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html as html_lib
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
import json
import re
from core.logging import get_logger
import mimetypes

from config.settings import Experiment
from core.errors import RetrievalError
from core.types import SearchResult
from retrieval.vqa import vqa_search, trake_search, enhanced_temporal_search
from retrieval.kis_detail_search import kis_detail_2stage_search
from retrieval.intelligent_search import intelligent_search
from retrieval.temporal_search import load_temporal_data
from retrieval.text_search import text_search
from retrieval.tracks import SUPPORTED_TRACKS, TrackQuery, build_retrieval_text
from indexing.readiness import read_readiness
from indexing.validation import verify_artifact_fingerprints, verify_frame_files
import numpy as np

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
    if status != "READY":
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

    # frame_path values in manifests are relative to the working directory the
    # pipeline was run from, not to experiment.run_dir (which can live outside
    # the project entirely, e.g. --runs-dir on another drive). Resolve against
    # cwd so /frame lookups work regardless of where runs are stored.
    set_project_root(Path.cwd())

    # Share one retriever (and its embedders) with the TRAKE / VQA / KIS Detail
    # code paths' own cache — two independently-built retrievers would each
    # hold a full copy of every configured embedder, which is what exhausted
    # VRAM here (two 3-model retrievers on a 4 GB GPU).
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


def _events_to_query(payload: dict) -> str:
    """Build a ``trake_search`` query string from a TRAKE payload.

    The UI posts ``events`` as a list of ``{text, sub_details}`` objects, while
    ``trake_search`` parses a multi-line ``E1: ...`` / ``E2: ...`` string. Fall
    back to a plain ``query`` field so direct API callers still work.
    """
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
                self._send_video_shots(parse_qs(parsed.query))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)

            # TRAKE: caller supplies each event. Enhanced: an LLM splits one
            # sentence into events first, then runs the identical pipeline.
            if parsed.path in ("/api/trake-search", "/api/enhanced-temporal-search"):
                try:
                    payload = self._read_json()
                    top_k = int(payload.get("top_k") or default_top_k)
                    window = int(payload.get("window", 15))
                    events_raw = payload.get("events")
                    if (
                        parsed.path == "/api/trake-search"
                        and isinstance(events_raw, list)
                        and len(events_raw) >= 2
                    ):
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
                        if parsed.path == "/api/enhanced-temporal-search":
                            result = enhanced_temporal_search(
                                query=str(payload.get("query", "")),
                                max_events=int(payload.get("max_events") or 5),
                                **shared,
                            )
                        else:
                            result = trake_search(query=_events_to_query(payload), **shared)

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
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("Temporal search failed (%s)", parsed.path)
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # VQA track uses full pipeline (Embedding search → Temporal → Agent)
            if parsed.path == "/api/vqa-search":
                try:
                    payload = self._read_json()
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
                    # Hydrate frame paths for image serving
                    for r in result.get("results", []):
                        if r.get("frame_path"):
                            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("VQA search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # Interactive search agent (AIC_2025-style narrowing loop) — stateless,
            # the frontend holds the conversation and re-sends it every turn.
            if parsed.path == "/api/agent/chat":
                try:
                    from agent.interactive import run_agent_turn

                    payload = self._read_json()
                    messages = payload.get("messages")
                    if not isinstance(messages, list) or not messages:
                        raise ValueError("messages (non-empty list) is required.")
                    result = run_agent_turn(messages, experiment)
                    for r in result.get("results", []):
                        if r.get("frame_path"):
                            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("Agent chat failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # ASR / OCR: direct BM25 text search, no embedding model involved.
            if parsed.path in ("/api/asr-search", "/api/ocr-search"):
                try:
                    payload = self._read_json()
                    query = str(payload.get("query", ""))
                    top_k = int(payload.get("top_k") or default_top_k)
                    source = "asr" if parsed.path == "/api/asr-search" else "ocr"

                    result = text_search(experiment, query=query, source=source, top_k=top_k)
                    for r in result.get("results", []):
                        if r.get("frame_path"):
                            r["image_url"] = f"/frame?path={quote(r['frame_path'])}"
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("Text search failed (%s)", parsed.path)
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # Intelligent: LLM splits the query into KIS/OCR/ASR + weights,
            # each enabled modality searches independently, fused by weighted SRRF.
            if parsed.path == "/api/intelligent-search":
                try:
                    payload = self._read_json()
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
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("Intelligent search failed")
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            # KIS Detail 2-Stage: coarse (general) → fine (specific)
            if parsed.path == "/api/kis-detail-2stage":
                try:
                    payload = self._read_json()
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
                    self._send_json(result)
                except Exception as exc:
                    LOGGER.exception("KIS Detail 2-Stage search failed")
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

                    frame_embeddings, frame_records = load_temporal_data(experiment)
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

        def _send_video_shots(self, query: dict) -> None:
            video_id = query.get("video_id", [""])[0]
            if not video_id:
                self._send_json({"error": "video_id required"}, status=HTTPStatus.BAD_REQUEST)
                return

            frames_path = experiment.run_dir / "manifests" / "frames.jsonl"
            experiment.run_dir / "manifests" / "shots.jsonl"

            try:
                # Group frames by shot_id from frames.jsonl
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

                self._send_json({"video_id": video_id, "shots": shot_list})
            except Exception:
                LOGGER.exception("Failed to load shots for video=%s", video_id)
                self._send_json({"video_id": video_id, "shots": []})

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
    header { padding: 18px 24px 12px; border-bottom: 1px solid var(--line); background: var(--panel); display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    #mode-switch { display: flex; gap: 6px; }
    .mode-btn { width: auto; margin: 0; padding: 6px 14px; font-size: 13px; background: transparent; color: var(--muted); border: 1px solid var(--line); border-radius: 6px; cursor: pointer; }
    .mode-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
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
    label.check {
      display: flex;
      align-items: center;
      gap: 7px;
      margin: 6px 0;
      color: var(--text);
      font-weight: 500;
      cursor: pointer;
    }
    label.check input { width: auto; margin: 0; }
    #model-config {
      margin: 10px 0 4px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fafafa;
    }
    #model-config > label:first-child { margin-top: 0; }
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
  <div><h1>CodeNova Retrieval UI</h1><small>Experiment: __ACTIVE_EXPERIMENT__ · Models: __ACTIVE_MODELS__</small></div>
  <div id="mode-switch">
    <button type="button" class="mode-btn active" data-mode="manual">🛠 Thủ công</button>
    <button type="button" class="mode-btn" data-mode="agent">🤖 Agent</button>
  </div>
</header>
<main>
  <aside>
    <form id="search-form">
      <label for="track">Retrieval Track</label>
      <select id="track" name="track">
        <optgroup label="KIS">
          <option value="textual_kis">KIS Basic</option>
          <option value="kis_detail_2stage">KIS Detail 2-Stage</option>
        </optgroup>
        <optgroup label="Text search">
          <option value="asr_search">ASR Search</option>
          <option value="ocr_search">OCR Search</option>
        </optgroup>
        <option value="intelligent">Intelligent (KIS+OCR+ASR)</option>
        <option value="vqa">VQA</option>
        <option value="trake">TRAKE</option>
        <option value="temporal_enhanced">Temporal Enhanced (LLM tách event)</option>
      </select>
      <div id="model-config">
        <label>Embedding models</label>
        <label class="check"><input type="checkbox" name="model_jina-clip-v2" checked> Jina-CLIP-v2</label>
        <label class="check"><input type="checkbox" name="model_siglip2-so400m" checked> SigLIP2</label>
        <label class="check"><input type="checkbox" name="model_vietnamese-embedding" checked> Vietnamese-Embedding</label>
        <label class="check"><input type="checkbox" id="use-reranker" checked> Rerank BLIP-2</label>
        <label class="check"><input type="checkbox" id="use-llm" checked> Enhance bằng Qwen</label>
      </div>
      <label for="query">Query</label>
      <textarea id="query" name="query">a person riding a motorbike</textarea>
      <label for="context">Scene / Context</label>
      <textarea id="context" name="context" placeholder="Optional shot sequence or scene description"></textarea>
      <label for="question">Question</label>
      <textarea id="question" name="question" placeholder="Use this for VQA or QA tracks"></textarea>
      <div id="events-section" style="display:none;">
        <label>Events / Scenes / Subqueries</label>
        <div id="events-list"></div>
        <button type="button" id="add-event-btn" style="width:auto;padding:6px 14px;margin-top:6px;font-size:13px;">+ Add</button>
        <div id="window-control" style="margin-top:10px;display:none;">
        <label for="window-slider">Temporal Window: <span id="window-value">15</span>s</label>
        <input id="window-slider" type="range" min="10" max="300" step="5" value="15" style="width:100%;margin-top:4px;">
        <div class="hint" style="margin-top:4px;font-size:12px;">Khoảng thời gian tối đa giữa 2 scene/event liền kề</div>
      </div>
      </div>
      <div id="kis-2stage-section" style="display:none;">
        <label>General Subqueries</label>
        <div id="general-events-list"></div>
        <button type="button" id="add-general-btn" style="width:auto;padding:6px 14px;margin-top:6px;font-size:13px;">+ Add general</button>
        <label style="margin-top:14px;">Specific Subqueries</label>
        <div id="specific-events-list"></div>
        <button type="button" id="add-specific-btn" style="width:auto;padding:6px 14px;margin-top:6px;font-size:13px;">+ Add specific</button>
      </div>
      <div class="row">
        <div>
          <label for="top-k">Top K</label>
          <input id="top-k" name="top_k" type="number" value="20" min="1" max="100">
        </div>
        <button id="submit" type="submit">Search</button>
        <div id="sidebar-answer" style="display:none; margin-top: 14px; padding: 12px 14px; border: 1px solid var(--accent); border-radius: 8px; background: #f0fdf8;">
          <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);">Answer</div>
          <div id="sidebar-answer-text" style="margin-top: 4px; font-size: 15px; font-weight: 700; color: var(--accent-strong); line-height: 1.4;"></div>
        </div>
      </form>
      <div id="agent-chat" style="display:none; margin-top: 16px; border-top: 1px solid #ddd; padding-top: 12px;">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);">Agent Chat</div>
        <div id="chat-messages" style="max-height: 220px; overflow-y: auto; font-size: 13px; margin: 8px 0; line-height: 1.45;"></div>
        <div id="chat-suggestions" style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px;"></div>
        <div style="display: flex; gap: 6px;">
          <input id="chat-input" type="text" placeholder="Mô tả cảnh cần tìm..." style="flex: 1; padding: 6px 8px; font-size: 13px;">
          <button id="chat-send" type="button">Gửi</button>
        </div>
      </div>
      <div id="status" class="status">Ready.</div>
    </aside>
    <section>
      <div id="answer-box"></div>
      <div id="events-box"></div>
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
    function eid(id) { return document.getElementById(id); }
    function esc(s) { return escapeHtml(s == null ? "" : String(s)); }
    function escapeHtml(s) {
      return String(s == null ? "" : s).replace(/[&<>"']/g, c => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
      ));
    }
    function formatTime(seconds) {
      const s = Math.max(0, Number(seconds) || 0);
      const mm = Math.floor(s / 60);
      const ss = Math.floor(s % 60);
      return `${mm}:${String(ss).padStart(2, "0")}`;
    }
    function fmtTime(seconds) { return formatTime(seconds); }
    function formatNumber(n) { return Number(n).toLocaleString(); }

    const form = document.getElementById("search-form");
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");
    const answerBox = document.getElementById("answer-box");
    const eventsEl = document.getElementById("events-box");
    const pipelineBox = document.getElementById("pipeline-box");
    const submitEl = document.getElementById("submit");
    const sidebarAnswer = document.getElementById("sidebar-answer");
    const sidebarAnswerText = document.getElementById("sidebar-answer-text");
    const EVENTS_LIST = eid("events-list");
    const EVENTS_SEC = eid("events-section");

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

  // ─── KIS Detail 2-Stage: general / specific event lists ──────────────────────
  const GEN_LIST = eid("general-events-list");
  const SPEC_LIST = eid("specific-events-list");

  function addGeneralEvent(value) {
    const wrapper = document.createElement("div");
    wrapper.className = "event-wrapper";
    wrapper.style.cssText = "margin-bottom:6px;";
    wrapper.innerHTML = `
      <div style="display:flex;gap:6px;align-items:start;">
        <textarea class="event-input" style="flex:1;min-height:48px;" placeholder="General subquery...">${esc(value||"")}</textarea>
        <button type="button" style="width:auto;padding:6px 10px;margin-top:0;font-size:14px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.closest('.event-wrapper').remove()" title="Remove">✕</button>
      </div>`;
    GEN_LIST.appendChild(wrapper);
  }

  function addSpecificEvent(value) {
    const wrapper = document.createElement("div");
    wrapper.className = "event-wrapper";
    wrapper.style.cssText = "margin-bottom:6px;";
    wrapper.innerHTML = `
      <div style="display:flex;gap:6px;align-items:start;">
        <textarea class="event-input" style="flex:1;min-height:48px;" placeholder="Specific subquery...">${esc(value||"")}</textarea>
        <button type="button" style="width:auto;padding:6px 10px;margin-top:0;font-size:14px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.closest('.event-wrapper').remove()" title="Remove">✕</button>
      </div>`;
    SPEC_LIST.appendChild(wrapper);
  }

  eid("add-general-btn").addEventListener("click", () => addGeneralEvent(""));
  eid("add-specific-btn").addEventListener("click", () => addSpecificEvent(""));

  function get2StageEvents() {
    const general = Array.from(GEN_LIST.querySelectorAll(".event-wrapper"))
      .map(w => w.querySelector(".event-input")?.value.trim()).filter(Boolean);
    const specific = Array.from(SPEC_LIST.querySelectorAll(".event-wrapper"))
      .map(w => w.querySelector(".event-input")?.value.trim()).filter(Boolean);
    return { general, specific };
  }

  // ─── Model picker (checkboxes named "model_<embedding-model-name>") ───────────
  function getEnabledModels() {
    return Array.from(document.querySelectorAll('#model-config input[name^="model_"]'))
      .filter(cb => cb.checked)
      .map(cb => cb.name.slice("model_".length));
  }

  // ─── Track selector ───────────────────────────────────────────────────────────
  form.track.addEventListener("change", () => {
    const t = form.track.value;
    const isBasic = t === "textual_kis";
    const is2Stage = t === "kis_detail_2stage";
    const isT = t === "trake";
    const isV = t === "vqa";
    const isTextSearch = t === "asr_search" || t === "ocr_search";
    const isIntelligent = t === "intelligent";
    const isEnhanced = t === "temporal_enhanced";
    const usesEvents = isT;
    // Enhanced temporal derives its own events from one sentence, so it keeps
    // the temporal window control but not the manual event list.
    const showWindow = isT || isEnhanced;
    // Only tracks that actually route through Retriever.search() (i.e. embed
    // the query with one or more configured models) show the model picker.
    const usesEmbedders = isBasic || isT || isV || isIntelligent || isEnhanced;

    eid("query").style.display = usesEvents || is2Stage ? "none" : "";
    form.querySelector("label[for=query]").style.display = usesEvents || is2Stage ? "none" : "";
    eid("context").style.display = isT || is2Stage || isTextSearch ? "none" : "";
    form.querySelector("label[for=context]").style.display = isT || is2Stage || isTextSearch ? "none" : "";
    eid("question").style.display = isV ? "" : "none";
    form.querySelector("label[for=question]").style.display = isV ? "" : "none";
    eid("top-k").style.display = isT || is2Stage ? "none" : "";
    form.querySelector("label[for=top-k]").style.display = isT || is2Stage ? "none" : "";
    EVENTS_SEC.style.display = usesEvents ? "" : "none";
    eid("window-control").style.display = showWindow ? "" : "none";
    eid("kis-2stage-section").style.display = is2Stage ? "" : "none";
    eid("model-config").style.display = usesEmbedders ? "" : "none";
    if (isT && eventCount()===0) { addEvent("a person riding a motorbike"); addEvent("a person falling off"); }
  });
  form.track.dispatchEvent(new Event("change"));

  // Last submitted events for TRAKE (carries sub-details)
  let lastTrakeInput = null;

  // ─── Search submit ────────────────────────────────────────────────────────────
  form.addEventListener("submit", async e => {
    e.preventDefault();
    submitEl.disabled = true;
    statusEl.className = "status"; statusEl.textContent = "Searching...";
    resultsEl.innerHTML = ""; answerBox.innerHTML = ""; pipelineBox.innerHTML = ""; eventsEl.innerHTML = "";
    sidebarAnswer.style.display = "none";
    const track = form.track.value;
    let endpoint, payload;
    if (track === "trake") {
      const events = getEvents();
      if (events.length < 2) { statusEl.className="status warn"; statusEl.textContent="Need at least 2 events."; submitEl.disabled=false; return; }
      lastTrakeInput = events;
      endpoint = "/api/trake-search";
      payload = {
        events,
        top_k: 300,
        window: Number(eid("window-slider").value),
        enabled_models: getEnabledModels(),
        use_reranker: eid("use-reranker").checked,
        use_llm: eid("use-llm").checked,
      };
    } else if (track === "kis_detail_2stage") {
      const { general, specific } = get2StageEvents();
      if (general.length < 1) { statusEl.className="status warn"; statusEl.textContent="Need at least 1 general subquery."; submitEl.disabled=false; return; }
      if (specific.length < 1) { statusEl.className="status warn"; statusEl.textContent="Need at least 1 specific subquery."; submitEl.disabled=false; return; }
      endpoint = "/api/kis-detail-2stage"; payload = { general, specific };
    } else if (track === "vqa") {
      endpoint = "/api/vqa-search";
      payload = { query: form.query.value, context: form.context.value, question: form.question.value, top_k: Number(form["top_k"].value||20) };
    } else if (track === "asr_search" || track === "ocr_search") {
      endpoint = track === "asr_search" ? "/api/asr-search" : "/api/ocr-search";
      payload = { query: form.query.value, top_k: 300 };
    } else if (track === "intelligent") {
      endpoint = "/api/intelligent-search";
      payload = { query: form.query.value, top_k: Number(form["top_k"].value||20) };
    } else if (track === "temporal_enhanced") {
      endpoint = "/api/enhanced-temporal-search";
      payload = {
        query: form.query.value, context: form.context.value,
        top_k: Number(form["top_k"].value||20), max_events: 5,
      };
    } else {
      endpoint = "/api/search";
      payload = {
        track, query: form.query.value, context: form.context.value, question: form.question.value,
        top_k: 300, enabled_models: getEnabledModels(), use_reranker: eid("use-reranker").checked,
        use_llm: eid("use-llm").checked,
      };
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || "Search failed");
      }

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
      } else if (track === "trake" || track === "temporal_enhanced") {
        if (data.videos) {
          const chains = data.videos || [];
          const uniqueVideos = new Set(chains.map(v => v.video_id)).size;
          statusEl.innerHTML = `<strong>${chains.length}</strong> chain(s) from <strong>${uniqueVideos}</strong> video(s) match all events <span class="pill">TRAKE</span>`;
          renderTrake(data);
        } else {
          const eventCount = (data.events || []).length;
          const extracted = data.extracted_events || [];
          const label = track === "trake" ? "TRAKE" : "Temporal Enhanced";
          const extractedNote = extracted.length
            ? ` · LLM tách: ${extracted.map(escapeHtml).join(" → ")}`
            : "";
          statusEl.innerHTML = `<strong>${eventCount}</strong> event(s) found <span class="pill">${label}</span>${extractedNote}`;
          renderTrakeEvents(data.events || []);
          renderPipeline(data);
          renderResults(data.results || []);
        }
      } else if (track === "kis_detail_2stage") {
        const results = data.results || [];
        statusEl.innerHTML = `<strong>${results.length}</strong> frames match all details <span class="pill">KIS Detail 2-Stage</span>`;
        renderResults(results);
      } else if (track === "intelligent") {
        const results = data.results || [];
        const a = data.analysis || {};
        const w = a.weights || {};
        const counts = data.component_counts || {};
        const parts = ["kis", "ocr", "asr"]
          .filter(k => (w[k] || 0) > 0)
          .map(k => `${k.toUpperCase()} ${(w[k]).toFixed(2)} (${counts[k] || 0} hit)`);
        statusEl.innerHTML = `<strong>${results.length}</strong> results <span class="pill">Intelligent</span>`
          + (parts.length ? ` · ${escapeHtml(parts.join(" + "))}` : "");
        renderResults(results);
      } else {
        const trackLabels = {
          textual_kis: "Textual KIS", asr_search: "ASR Search", ocr_search: "OCR Search",
        };
        const trackLabel = trackLabels[track] || track;
        statusEl.innerHTML = `<strong>${data.results.length}</strong> results for <span class="pill">${trackLabel}</span>`;
        renderResults(data.results);
      }
    } catch (error) {
      statusEl.className = "status warn";
      statusEl.textContent = error.message;
    } finally {
      submitEl.disabled = false;
    }
  });

    function renderAnswer(answer) {
      answerBox.innerHTML = `
        <div class="answer-box">
          <div class="label">Answer</div>
          <div class="answer-text">${escapeHtml(answer)}</div>
        </div>`;
    }

    function renderPipeline(data) {
      const pipeline = data.pipeline || {};
      const hasAgent = pipeline.agent;
      const stages = hasAgent ? [
        { key: "embed_search", label: "Embedding Search", desc: `Top-${pipeline.embed_search?.top_k} frames retrieved` },
        { key: "temporal_search", label: "Temporal Search", desc: `${pipeline.temporal_search?.segments_found || 0} segments found` },
        { key: "gather_shot", label: "Shot Gather", desc: `${pipeline.gather_shot?.shots_count || 0} valid shots gathered` },
        { key: "shot_validation", label: "Shot Validation", desc: `Score: ${(pipeline.shot_validation?.validation_score || 0).toFixed(4)}` },
        { key: "agent", label: "Agent (Qwen3.5-4B)", desc: `Answer: ${(pipeline.agent?.answer || "N/A").substring(0, 100)}` },
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

  const thumbState = {};
  function trakeKey(videoId, chainIdx, eventIdx) { return videoId + "::" + chainIdx + "::" + eventIdx; }

  function renderTrake(data) {
    const videos = data.videos || [];
    if (!videos.length) {
      resultsEl.innerHTML = `<div class="hint" style="padding:20px;text-align:center;">No video found matching all events.</div>`;
      return;
    }
    resultsEl.innerHTML = videos.map((video, vi) => {
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
            <img src="${escapeHtml(url)}" alt="Event ${ei+1}" loading="lazy"
              id="evimg-${safeKey}"
              onclick="openModalFromCard('${safeKey}')">
            <button class="revert-badge" onclick="revertCard('${escapeHtml(video.video_id)}',${vi},${ei})">&#x21a9; Revert</button>
            <div class="ev-info">
              <span id="evinfo-text-${safeKey}"><strong>Event ${ei+1}</strong> &middot; rank ${st.rank} &middot; ${formatTime(st.currentTimestamp)}</span>
              ${isCustom?`<span class="pill" style="font-size:10px">CUSTOM</span>`:""}
            </div>
          </div>`;
      }).join("");
      return `
        <div class="video-block">
          <div class="video-block-header">
            <strong style="font-size:15px">#${vi+1} ${escapeHtml(video.video_name||video.video_id)}</strong>
            <span class="pill">Score: ${video.score} ${video.temporal_order_valid?"✓ temporal":"✗ temporal"}</span>
          </div>
          <div class="event-grid" style="grid-template-columns:repeat(${cols},1fr)">${evHtml}</div>
        </div>`;
    }).join("");
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
      if (!results || !results.length) {
        resultsEl.innerHTML = `<div class="hint" style="padding:20px;text-align:center;">No matching results found.</div>`;
        return;
      }
      resultsEl.innerHTML = results.map((result, index) => {
        // Sub-scores for 2-stage or multi-query
        const subs = result.sub_scores || result.subquery_scores || {};
        const subKeys = Object.keys(subs);
        const subBadgesHtml = subKeys.length > 0
          ? `<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;">${subKeys.map(k => `<span class="pill" style="font-size:10px;background:#eef2ff;color:#4338ca;">${escapeHtml(k.replace('sub_',''))}: ${Number(subs[k]).toFixed(3)}</span>`).join('')}</div>`
          : "";

        // Component scores for Intelligent fusion
        const compHtml = (result.kis_score || result.ocr_score || result.asr_score)
          ? `<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;">
              ${result.kis_score ? `<span class="pill" style="font-size:10px;background:#e0f2fe;color:#0369a1;">KIS ${Number(result.kis_score).toFixed(3)}</span>` : ""}
              ${result.ocr_score ? `<span class="pill" style="font-size:10px;background:#fef3c7;color:#b45309;">OCR ${Number(result.ocr_score).toFixed(3)}</span>` : ""}
              ${result.asr_score ? `<span class="pill" style="font-size:10px;background:#f3e8ff;color:#6b21a8;">ASR ${Number(result.asr_score).toFixed(3)}</span>` : ""}
            </div>`
          : "";

        // Source badge
        const srcBadge = result.source
          ? `<span class="pill" style="font-size:10px;margin-left:6px;">${escapeHtml(result.source.toUpperCase())}</span>`
          : "";

        // Text snippet
        const textContent = result.text || result.matched_text || result.ocr_text || result.asr_text || "";
        const textHtml = textContent
          ? `<div style="margin-top:6px;padding:4px 8px;font-size:12px;background:#f8fafc;border-left:3px solid var(--accent);border-radius:4px;color:var(--text);">${escapeHtml(textContent.slice(0, 160))}</div>`
          : "";

        return `
        <article class="card">
          <img src="${escapeHtml(result.image_url || "")}" alt="Result frame ${index + 1}" loading="lazy"
            onclick="openModal('${escapeHtml(result.image_url || "")}','${escapeHtml(result.video_id || "")}','${escapeHtml(result.frame_id || "")}',null,null)">
          <div class="meta">
            <div><strong>#${index + 1}</strong> score ${Number(result.score).toFixed(4)} ${srcBadge}</div>
            <div>${formatTime(result.timestamp_sec)} · frame ${formatNumber(result.frame_index)}</div>
            <div><strong>${escapeHtml(result.video_name || result.video_id || "")}</strong></div>
            <div>shot ${escapeHtml(result.shot_id || "s0")}</div>
            <code>${escapeHtml(result.frame_id || "")}</code>
            ${subBadgesHtml}
            ${compHtml}
            ${textHtml}
          </div>
        </article>`;
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
    statusEl.innerHTML = `<strong>Reverted</strong> Event ${eventIdx+1} to original <span class="pill">ORIGINAL</span>`;
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

    // Nav buttons disabled status
    const isFirst = modal.shotIdx <= 0 && modal.frameIdx <= 0;
    const currentShotFrames = shot.frames || [];
    const isLast = modal.shotIdx >= modal.shots.length - 1 && modal.frameIdx >= currentShotFrames.length - 1;
    eid("modal-prev").disabled = isFirst;
    eid("modal-next").disabled = isLast;

    // Thumbnail buttons (only for TRAKE)
    if (modal.eventIdx !== null && modal.eventIdx !== undefined) {
      const key = trakeKey(modal.videoId, modal.chainIdx, modal.eventIdx);
      const st = thumbState[key];
      const isCustom = st && st.currentUrl !== st.originalUrl;
      eid("btn-setthumb").textContent = isCustom ? "Cập nhật thumbnail" : "Làm thumbnail";
      eid("btn-revert-modal").style.display = isCustom ? "inline-block" : "none";
    }
  }

  // ─── Continuous Frame Navigation (Next / Previous Frame) ──────────────────────
  function prevFrame() {
    if (!modal.shots.length) return;
    if (modal.frameIdx > 0) {
      modal.frameIdx--;
    } else if (modal.shotIdx > 0) {
      modal.shotIdx--;
      const prevShotFrames = modal.shots[modal.shotIdx]?.frames || [];
      modal.frameIdx = Math.max(0, prevShotFrames.length - 1);
    }
    renderModal();
  }

  function nextFrame() {
    if (!modal.shots.length) return;
    const currentShotFrames = modal.shots[modal.shotIdx]?.frames || [];
    if (modal.frameIdx < currentShotFrames.length - 1) {
      modal.frameIdx++;
    } else if (modal.shotIdx < modal.shots.length - 1) {
      modal.shotIdx++;
      modal.frameIdx = 0;
    }
    renderModal();
  }

  // ─── Strip click ─────────────────────────────────────────────────────────────
  function selectStrip(e, fi) {
    if (e && e.stopPropagation) e.stopPropagation();
    modal.frameIdx = fi;
    renderModal();
  }

  // ─── Navigation button click handlers ───────────────────────────────────────
  eid("modal-prev").addEventListener("click", e => {
    e.stopPropagation();
    prevFrame();
  });
  eid("modal-next").addEventListener("click", e => {
    e.stopPropagation();
    nextFrame();
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
    renderModal();
    statusEl.innerHTML = `<strong>Thumbnail updated</strong> — Event ${modal.eventIdx+1} <span class="pill">CUSTOM</span>`;
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
    if (e.key === "ArrowLeft") { e.preventDefault(); prevFrame(); }
    if (e.key === "ArrowRight") { e.preventDefault(); nextFrame(); }
    if (e.key === "ArrowUp" && modal.shotIdx > 0) { e.preventDefault(); modal.shotIdx--; modal.frameIdx = 0; renderModal(); }
    if (e.key === "ArrowDown" && modal.shotIdx < modal.shots.length - 1) { e.preventDefault(); modal.shotIdx++; modal.frameIdx = 0; renderModal(); }
  });

  // Toggle VQA settings visibility
  form.track.addEventListener('change', (e) => {
      document.getElementById('vqa-settings').style.display = e.target.value === 'vqa' ? 'block' : 'none';
    });
    // Trigger initially
    if (form.track.value === 'vqa') {
        document.getElementById('vqa-settings').style.display = 'block';
    }

    // --- Mode switch: manual search vs agent chat ---
    function setMode(mode) {
      const isAgent = mode === "agent";
      document.getElementById("agent-chat").style.display = isAgent ? "" : "none";
      document.getElementById("search-form").style.display = isAgent ? "none" : "";
      document.querySelectorAll(".mode-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
      });
      localStorage.setItem("ui_mode", mode);
    }
    document.querySelectorAll(".mode-btn").forEach(btn => {
      btn.addEventListener("click", () => setMode(btn.dataset.mode));
    });
    setMode(localStorage.getItem("ui_mode") || "manual");

    // --- Interactive agent chat (stateless server: we keep the history) ---
    const chatMessages = [];
    const chatBox = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const chatSend = document.getElementById("chat-send");
    const chatSuggestions = document.getElementById("chat-suggestions");

    function chatRender() {
      chatBox.innerHTML = chatMessages.map(m =>
        `<div style="margin-bottom:6px;"><b>${m.role === "user" ? "Bạn" : "Agent"}:</b> ${escapeHtml(m.content)}</div>`
      ).join("");
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function chatSubmit(text) {
      if (!text.trim()) return;
      chatMessages.push({ role: "user", content: text.trim() });
      chatInput.value = "";
      chatSuggestions.innerHTML = "";
      chatRender();
      chatSend.disabled = true;
      statusEl.textContent = "Agent đang tìm...";
      try {
        const resp = await fetch("/api/agent/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: chatMessages }),
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        chatMessages.push({ role: "assistant", content: data.message || "" });
        chatRender();
        (data.suggestions || []).forEach(s => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.textContent = s;
          btn.style.cssText = "font-size:12px;padding:3px 8px;";
          btn.onclick = () => chatSubmit(s);
          chatSuggestions.appendChild(btn);
        });
        if (data.results && data.results.length) renderResults(data.results);
        statusEl.textContent = `Agent: ${data.results ? data.results.length : 0} kết quả.`;
      } catch (err) {
        statusEl.textContent = "Agent lỗi: " + err.message;
      } finally {
        chatSend.disabled = false;
      }
    }

    chatSend.addEventListener("click", () => chatSubmit(chatInput.value));
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); chatSubmit(chatInput.value); }
    });
  </script>
</body>
</html>
"""
