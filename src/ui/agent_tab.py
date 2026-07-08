"""Agent tab UI and lightweight retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote
from uuid import uuid4
import re

from config.settings import Experiment
from core.types import SearchResult
from retrieval.vqa import trake_search, vqa_search


@dataclass
class AgentSessionState:
    """Conversation state kept in memory for one browser session."""

    turns: list[dict[str, str]] = field(default_factory=list)
    pending_route: str = ""


AGENT_TAB_STYLE = r"""
    .mode-tabs { display: flex; gap: 8px; margin-bottom: 14px; }
    .mode-tab {
      flex: 1; margin-top: 0; padding: 10px 12px; border-radius: 10px;
      border: 1px solid var(--line); background: #f8faf9; color: var(--muted);
      font-weight: 750; cursor: pointer;
    }
    .mode-tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
    .agent-panel {
      border: 1px solid var(--line); border-radius: 12px; background: linear-gradient(180deg, #ffffff, #f8fbfa);
      padding: 14px;
    }
    .agent-panel-head {
      display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px;
    }
    .agent-kicker {
      color: var(--accent-strong); font-size: 12px; font-weight: 800; text-transform: uppercase;
      letter-spacing: .08em; margin-bottom: 4px;
    }
    .agent-title { font-size: 14px; color: var(--text); line-height: 1.45; max-width: 320px; }
    .agent-status-pill {
      padding: 6px 10px; border-radius: 999px; background: #e6f5f3; color: var(--accent-strong);
      font-size: 12px; font-weight: 750; white-space: nowrap;
    }
    .agent-thread {
      display: grid; gap: 10px; min-height: 180px; max-height: 340px; overflow: auto;
      padding: 8px; border: 1px solid var(--line); border-radius: 10px; background: #fff;
    }
    .agent-empty {
      display: grid; place-items: center; padding: 22px 16px; color: var(--muted); text-align: center;
      font-size: 13px; line-height: 1.5;
    }
    .agent-bubble {
      max-width: 100%; border-radius: 12px; padding: 10px 12px; line-height: 1.45; font-size: 13px;
      border: 1px solid var(--line); background: #f9fafb;
    }
    .agent-bubble.user { margin-left: auto; background: #ecfdf5; border-color: #c7ebd8; }
    .agent-bubble.assistant { background: #fff; }
    .agent-bubble .meta {
      display: flex; justify-content: space-between; gap: 8px; margin-bottom: 4px; font-size: 11px; color: var(--muted);
      text-transform: uppercase; letter-spacing: .05em;
    }
    .agent-bubble .text { white-space: pre-wrap; }
    .agent-trace { margin-top: 8px; }
    .agent-trace summary { cursor: pointer; color: var(--accent-strong); font-size: 12px; font-weight: 700; }
    .agent-trace pre {
      white-space: pre-wrap; margin: 8px 0 0; padding: 10px 11px; border-radius: 8px;
      background: #f8fafc; border: 1px solid var(--line); color: #334155; font-size: 12px; line-height: 1.45;
    }
    .agent-compose { margin-top: 12px; }
    .agent-compose textarea { min-height: 90px; }
    .agent-compose-actions { display: flex; gap: 8px; }
    .agent-compose-actions button { width: auto; margin-top: 10px; }
    .agent-secondary {
      background: #fff; color: var(--text); border-color: var(--line);
    }
    .agent-secondary:hover { background: #f8faf9; }
"""


AGENT_TAB_HTML = r"""
    <div id="agent-panel" style="display:none;">
      <div class="agent-panel">
        <div class="agent-panel-head">
          <div>
            <div class="agent-kicker">Agent</div>
            <div class="agent-title">Auto-route a query to KIS, TRAKE, VQA, or ad-hoc retrieval.</div>
          </div>
          <div id="agent-status" class="agent-status-pill">Idle</div>
        </div>
        <div id="agent-thread" class="agent-thread">
          <div class="agent-empty">
            Start a conversation and the agent will decide which retrieval path to use.
          </div>
        </div>
        <div class="agent-compose">
          <label for="agent-input">Message</label>
          <textarea id="agent-input" placeholder="Ask for KIS, TRAKE, VQA, or anything ad-hoc."></textarea>
          <div class="agent-compose-actions">
            <button type="button" id="agent-send">Send</button>
            <button type="button" id="agent-clear" class="agent-secondary">Clear</button>
          </div>
        </div>
      </div>
    </div>
"""


AGENT_TAB_SCRIPT = r"""
    <script>
      (function () {
        const STORAGE_KEY = "codenova-agent-session-id";
        const SESSION_ID = sessionStorage.getItem(STORAGE_KEY) || (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()));
        sessionStorage.setItem(STORAGE_KEY, SESSION_ID);

        const THREAD = document.getElementById("agent-thread");
        const INPUT = document.getElementById("agent-input");
        const SEND = document.getElementById("agent-send");
        const CLEAR = document.getElementById("agent-clear");
        const STATUS = document.getElementById("agent-status");

        const state = {
          sessionId: SESSION_ID,
          busy: false,
          turns: [],
        };

        function esc(v) {
          return String(v)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
        }

        function setStatus(text, warn) {
          if (!STATUS) return;
          STATUS.textContent = text;
          STATUS.style.background = warn ? "#fff7ed" : "#e6f5f3";
          STATUS.style.color = warn ? "#a16207" : "var(--accent-strong)";
        }

        function renderThread() {
          if (!THREAD) return;
          if (!state.turns.length) {
            THREAD.innerHTML = '<div class="agent-empty">Start a conversation and the agent will decide which retrieval path to use.</div>';
            return;
          }
          THREAD.innerHTML = state.turns.map(turn => {
            const trace = Array.isArray(turn.trace) && turn.trace.length
              ? `<details class="agent-trace"><summary>Trace</summary><pre>${esc(turn.trace.map(step => `${step.step}: ${step.detail}`).join("\n"))}</pre></details>`
              : "";
            return `
              <div class="agent-bubble ${esc(turn.role)}">
                <div class="meta"><span>${esc(turn.role)}</span><span>${esc(turn.route || "")}</span></div>
                <div class="text">${esc(turn.text || "")}</div>
                ${turn.reply ? `<div class="text" style="margin-top:8px;color:var(--accent-strong);font-weight:700;">${esc(turn.reply)}</div>` : ""}
                ${trace}
              </div>`;
          }).join("");
          THREAD.scrollTop = THREAD.scrollHeight;
        }

        function renderEvidence(data) {
          if (!data) return;
          if (typeof window.renderTrake === "function" && data.videos) {
            window.renderTrake(data);
            return;
          }
          if (typeof window.renderCards === "function" && Array.isArray(data.results)) {
            window.renderCards(data.results);
          }
        }

        async function submitMessage() {
          const message = INPUT ? INPUT.value.trim() : "";
          if (!message || state.busy) return;

          state.busy = true;
          if (SEND) SEND.disabled = true;
          if (CLEAR) CLEAR.disabled = true;
          setStatus("Thinking...");

          state.turns.push({ role: "user", text: message });
          renderThread();
          if (INPUT) INPUT.value = "";

          try {
            const res = await fetch("/api/agent-chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ session_id: state.sessionId, message }),
            });
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || "Agent request failed");

            state.turns.push({
              role: "assistant",
              text: data.reply || data.answer || "No reply generated.",
              reply: data.route ? `Route: ${data.route}` : "",
              route: data.route || "",
              trace: data.trace || [],
            });
            renderThread();
            renderEvidence(data);

            if (data.needs_follow_up) {
              setStatus(data.follow_up || "Need more detail.", true);
            } else {
              setStatus(data.route ? `Answered via ${data.route}` : "Answered");
            }
          } catch (err) {
            state.turns.push({ role: "assistant", text: err.message, route: "error" });
            renderThread();
            setStatus(err.message, true);
          } finally {
            state.busy = false;
            if (SEND) SEND.disabled = false;
            if (CLEAR) CLEAR.disabled = false;
          }
        }

        if (SEND) SEND.addEventListener("click", submitMessage);
        if (CLEAR) {
          CLEAR.addEventListener("click", () => {
            state.turns = [];
            renderThread();
            setStatus("Idle");
            if (typeof window.clearResults === "function") {
              window.clearResults();
            }
          });
        }
        if (INPUT) {
          INPUT.addEventListener("keydown", event => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submitMessage();
            }
          });
        }

        setStatus("Idle");
        renderThread();
      })();
    </script>
"""


def build_agent_payload(
    experiment: Experiment,
    retriever,
    payload: dict[str, object],
    sessions: dict[str, AgentSessionState],
    default_top_k: int,
) -> dict[str, object]:
    """Handle a conversational agent request with lightweight routing."""

    session_id = str(payload.get("session_id") or uuid4())
    message = str(payload.get("message", "")).strip()
    context = str(payload.get("context", "")).strip()
    top_k = int(payload.get("top_k") or default_top_k)

    if not message:
        return {
            "session_id": session_id,
            "route": "error",
            "reply": "Message is empty.",
            "trace": [{"step": "input", "detail": "No message provided."}],
            "results": [],
            "needs_follow_up": False,
        }

    state = sessions.setdefault(session_id, AgentSessionState())
    state.turns.append({"role": "user", "text": message})

    history_text = " ".join(turn["text"] for turn in state.turns[-4:] if turn["role"] == "user")
    route, reason, needs_follow_up, follow_up = _classify_route(message, history_text)
    trace: list[dict[str, str]] = [{"step": "router", "detail": reason}]

    if needs_follow_up:
        reply = follow_up
        state.turns.append({"role": "assistant", "text": reply})
        return {
            "session_id": session_id,
            "route": route,
            "reply": reply,
            "trace": trace,
            "results": [],
            "needs_follow_up": True,
            "follow_up": follow_up,
        }

    if route == "trake":
        events = _extract_trake_events(message)
        if len(events) < 2:
            reply = "Please provide at least 2 event descriptions for TRAKE, for example: 'person enters room; person leaves room'."
            trace.append({"step": "clarify", "detail": "Not enough event descriptions for TRAKE."})
            state.turns.append({"role": "assistant", "text": reply})
            return {
                "session_id": session_id,
                "route": route,
                "reply": reply,
                "trace": trace,
                "results": [],
                "needs_follow_up": True,
                "follow_up": reply,
            }

        result = trake_search(experiment=experiment, events=events, top_k=top_k)
        trace.append({"step": "retrieval", "detail": f"TRAKE searched {len(events)} event(s)."})
        reply = _summarize_trake(result)
        trace.append({"step": "answer", "detail": reply[:240]})
        state.turns.append({"role": "assistant", "text": reply})
        return {
            "session_id": session_id,
            "route": route,
            "reply": reply,
            "trace": trace,
            "videos": result.get("videos", []),
            "results": _flatten_trake_results(result),
            "needs_follow_up": False,
        }

    if route == "vqa":
        result = vqa_search(
            experiment=experiment,
            query=message,
            question=message,
            context=context,
            top_k=top_k,
        )
        trace.append({"step": "retrieval", "detail": f"VQA pipeline returned {len(result.get('results', []))} result(s)."})
        reply = str(result.get("answer", "No answer generated.")).strip()
        trace.append({"step": "answer", "detail": reply[:240]})
        state.turns.append({"role": "assistant", "text": reply})
        return {
            "session_id": session_id,
            "route": route,
            "reply": reply,
            "trace": trace,
            "results": result.get("results", []),
            "pipeline": result.get("pipeline", {}),
            "needs_follow_up": False,
        }

    retrieval_text = _build_retrieval_text(route, message, context, history_text)
    results = retriever.search(query=retrieval_text, top_k=top_k)
    trace.append({"step": "retrieval", "detail": f"Vector search on '{retrieval_text}'."})
    reply = _summarize_results(route, message, results)
    trace.append({"step": "answer", "detail": reply[:240]})
    state.turns.append({"role": "assistant", "text": reply})

    return {
        "session_id": session_id,
        "route": route,
        "reply": reply,
        "trace": trace,
        "results": [result_to_payload(result) for result in results],
        "needs_follow_up": False,
    }


def _classify_route(message: str, history_text: str) -> tuple[str, str, bool, str]:
    text = f"{history_text} {message}".strip().lower()
    tokens = len(message.split())
    has_question = "?" in message or any(word in text for word in ("what", "who", "where", "when", "how", "màu", "color"))
    has_trake = any(word in text for word in ("trake", "event", "events", "then", "after", "before"))
    has_vqa = any(word in text for word in ("vqa", "question", "answer", "describe", "caption")) or has_question
    has_video = any(word in text for word in ("video", "clip", "sample video", "video similar"))

    if tokens < 3 and not has_trake and not has_vqa and not has_video:
        return (
            "ad_hoc",
            "Query is too short to auto-route confidently.",
            True,
            "Please add a bit more detail, or tell me whether this is KIS, TRAKE, or VQA.",
        )

    if has_trake:
        return (
            "trake",
            "Detected multi-event wording, so the request looks like TRAKE.",
            False,
            "",
        )
    if has_vqa:
        return (
            "vqa",
            "Detected a question-oriented query, so the request looks like VQA.",
            False,
            "",
        )
    if has_video:
        return (
            "video_kis",
            "Detected video-sample wording, so the request looks like Video KIS.",
            False,
            "",
        )

    return (
        "textual_kis",
        "Cannot classify the request. Defaulted to textual KIS.",
        False,
        "",
    )


def _build_retrieval_text(route: str, message: str, context: str, history_text: str) -> str:
    parts = [context, history_text, message]
    if route == "ad_hoc":
        parts = [message, context, history_text]
    return " ".join(part.strip() for part in parts if part and part.strip())


def _extract_trake_events(message: str) -> list[str]:
    raw_parts = re.split(r"\n+|\s*->\s*|\s*;\s*|\s+then\s+|\s+and\s+then\s+", message, flags=re.IGNORECASE)
    events = [part.strip(" -:\t") for part in raw_parts if part and part.strip(" -:\t")]
    return events


def _summarize_results(route: str, message: str, results: list[SearchResult]) -> str:
    if not results:
        return f"I couldn't find a strong match for: {message}"
    best = results[0]
    return (
        f"I routed this as {route.replace('_', ' ')} and found {len(results)} candidate(s). "
        f"Best match: {best.video_name or best.video_id} (score {best.score:.4f})."
    )


def _summarize_trake(result: dict[str, Any]) -> str:
    videos = result.get("videos", [])
    if not videos:
        return "No TRAKE chain matched all events."
    top = videos[0]
    return (
        f"Found {len(videos)} TRAKE chain(s). Best video: {top.get('video_name') or top.get('video_id')} "
        f"with score {top.get('score', 0)}."
    )


def _flatten_trake_results(result: dict[str, Any]) -> list[dict[str, object]]:
    flat: list[dict[str, object]] = []
    for video in result.get("videos", []):
        for event in video.get("events", []):
            item = {
                "video_id": video.get("video_id", ""),
                "video_name": video.get("video_name", ""),
                "score": float(video.get("score", 0.0)),
                "frame_id": event.get("frame_id", ""),
                "frame_path": event.get("frame_path", ""),
                "timestamp_sec": event.get("timestamp_sec"),
                "frame_index": event.get("frame_index"),
                "shot_id": event.get("shot_id", ""),
            }
            if event.get("frame_path"):
                item["image_url"] = f"/frame?path={quote(str(event['frame_path']))}"
            flat.append(item)
    return flat


def result_to_payload(result: SearchResult) -> dict[str, object]:
    payload = result.to_dict()
    if result.frame_path:
      payload["image_url"] = f"/frame?path={quote(str(result.frame_path))}"
    return payload