"""Interactive search agent — the AIC_2025 narrowing-loop flow, Docker-served.

Ported from the AIC_2025 reference project's ``online/backend/agent.py``: the
LLM holds five tools (search_kis / search_asr / search_ocr /
subagent_summarize / ask_user), runs up to ``MAX_TOOL_ROUNDS`` tool calls per
turn, and always ends a turn either with a plain message or an ``ask_user``
question that narrows the search. Stateless — the frontend keeps the
conversation and re-sends it every turn.

Differences from the reference (see ``prompts/agent.py`` for why): tool calls
are JSON-in-text rather than the OpenAI ``tools`` API, and the backend is the
Docker-hosted Qwen3.5-4B on port 8888 instead of GPT-4o.
"""

from __future__ import annotations

import json
import logging
import re

from config.settings import Experiment
from prompts.agent import INTERACTIVE_SYSTEM_PROMPT, SUBAGENT_PROMPT

LOGGER = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6  # số vòng gọi tool tối đa trong 1 lượt, tránh loop vô hạn
MAX_RESULTS_TO_MODEL = 12  # số kết quả tóm tắt cho model đọc (tiết kiệm token)

_EXHAUSTED_MESSAGE = (
    "Mình đã chạy tối đa số vòng tìm kiếm cho lượt này. "
    "Bạn xem các kết quả hiện tại hoặc bổ sung thêm chi tiết giúp mình nhé."
)


class InteractiveAgent:
    """One experiment's interactive agent; caches retriever/text-index/captions."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self._client = None
        self._retriever = None
        self._text_index = None
        self._captions: dict[str, str] | None = None

    # -- lazy resources -----------------------------------------------------

    def _load_client(self):
        if self._client is None:
            import os

            from agent.hardware import default_agent_model
            from modules._vllm_chat import VllmChatClient

            self._client = VllmChatClient(
                base_url=os.environ.get("AGENT_LOCAL_ENGINE_URL", "http://localhost:8888/v1"),
                model_name=default_agent_model(),
            )
        return self._client

    def _load_retriever(self):
        if self._retriever is None:
            from retrieval import build_retriever

            self._retriever = build_retriever(self.experiment)
        return self._retriever

    def _load_text_index(self):
        if self._text_index is None:
            from stores.text.factory import build_text_index

            self._text_index = build_text_index(self.experiment)
        return self._text_index

    def _load_captions(self) -> dict[str, str]:
        if self._captions is None:
            from repository.caption_repo import CaptionRepository

            try:
                self._captions = CaptionRepository(self.experiment).by_id()
            except Exception:
                LOGGER.exception("Caption manifest unavailable")
                self._captions = {}
        return self._captions

    # -- turn loop ----------------------------------------------------------

    def run_turn(self, messages: list[dict]) -> dict:
        """Run one agent turn over the client-held conversation.

        Returns ``{message, results, actions, question, suggestions, done}``.
        ``done=False`` means the turn ended on an ask_user question.
        """
        convo = [
            {"role": m["role"], "content": str(m.get("content", ""))}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        results: list[dict] = []
        actions: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                raw = self._load_client().complete_text(
                    system_prompt=INTERACTIVE_SYSTEM_PROMPT,
                    user_prompt="",
                    extra_messages=convo,
                    generation_params={"temperature": 0.2, "max_tokens": 512},
                )
            except Exception as exc:
                LOGGER.exception("Interactive agent LLM call failed")
                return _turn(
                    f"Agent LLM chưa sẵn sàng ({exc}). Chạy `make agent-up` rồi thử lại.",
                    results,
                    actions,
                    done=True,
                )

            data = _parse_tool_json(raw)
            tool = data.get("tool") if data else None

            if not tool:
                message = (data or {}).get("message") or raw.strip()
                return _turn(message, results, actions, done=True)

            args = data.get("args") or {}
            if tool == "ask_user":
                question = (
                    str(args.get("question", "")).strip()
                    or "Bạn mô tả thêm chi tiết giúp mình nhé?"
                )
                suggestions = [str(s) for s in args.get("suggestions", [])][:4]
                actions.append({"tool": tool, "args": args})
                return _turn(
                    question,
                    results,
                    actions,
                    done=False,
                    question=question,
                    suggestions=suggestions,
                )

            observation, results = self._run_tool(tool, args, results)
            actions.append({"tool": tool, "args": args})
            convo.append({"role": "assistant", "content": raw.strip()})
            convo.append({"role": "user", "content": f"[KẾT QUẢ TOOL {tool}]\n{observation}"})

        return _turn(_EXHAUSTED_MESSAGE, results, actions, done=True)

    # -- tools --------------------------------------------------------------

    def _run_tool(self, tool: str, args: dict, results: list[dict]) -> tuple[str, list[dict]]:
        """Execute one tool; returns (observation-for-model, updated results)."""
        try:
            if tool == "search_kis":
                query = str(args.get("query", "")).strip()
                if not query:
                    return "Thiếu tham số query.", results
                top_k = int(args.get("num_results", 40))
                hits = self._load_retriever().search(query=query, top_k=top_k)
                results = [r.to_dict() for r in hits]
                return _summarize_for_model(results), results

            if tool in ("search_asr", "search_ocr"):
                query = str(args.get("query", "")).strip()
                if not query:
                    return "Thiếu tham số query.", results
                top_k = int(args.get("num_results", 20))
                source = "asr" if tool == "search_asr" else "ocr"
                try:
                    docs = self._load_text_index().search_documents(query, top_k, source=source)
                except Exception as exc:
                    return (
                        f"Chỉ mục văn bản ({source}) chưa sẵn sàng: {exc}. "
                        "Hãy dùng search_kis thay thế.",
                        results,
                    )
                if not docs:
                    return f"Không có kết quả {source} nào — thử search_kis.", results
                results = [
                    {
                        "frame_id": d.get("frame_id"),
                        "video_id": d.get("video_id"),
                        "video_name": d.get("video_id"),
                        "timestamp_sec": d.get("timestamp_sec"),
                        "score": d.get("score", 0.0),
                        "text": d.get("text", ""),
                    }
                    for d in docs
                ]
                return _summarize_for_model(results), results

            if tool == "subagent_summarize":
                focus = str(args.get("focus", "")).strip()
                return self._subagent_summarize(focus, results), results

            return f"Tool không tồn tại: {tool}", results
        except Exception as exc:
            LOGGER.exception("Tool %s failed", tool)
            return f"[Tool {tool} lỗi: {exc}]", results

    def _subagent_summarize(self, focus: str, results: list[dict]) -> str:
        """Trợ lý phụ: đọc caption của kết quả hiện tại → tổng hợp nhóm cảnh."""
        if not results:
            return "Chưa có kết quả nào để tổng hợp — hãy chạy search trước."
        captions = self._load_captions()
        lines = []
        for r in results[:40]:
            cap = r.get("caption") or captions.get(str(r.get("frame_id")), "")
            if not cap:
                continue
            video = r.get("video_name") or r.get("video_id") or "?"
            ts = r.get("timestamp_sec")
            lines.append(f"- [{video} @ {ts}s] {cap}")
        if not lines:
            return (
                "Kết quả hiện tại không có caption (experiment chưa chạy captioning) — "
                "hãy hỏi người dùng dựa trên phân bố video/timestamp."
            )
        user_prompt = (
            f"Người dùng đang tìm: {focus or 'không rõ'}\n"
            f"Caption của các keyframe:\n" + "\n".join(lines)
        )
        return self._load_client().complete_text(
            system_prompt=SUBAGENT_PROMPT,
            user_prompt=user_prompt,
            generation_params={"temperature": 0.2, "max_tokens": 400},
        )


def _summarize_for_model(results: list[dict]) -> str:
    """Rút gọn kết quả thành text ngắn cho model đọc (không gửi payload đầy đủ)."""
    if not results:
        return "Không có kết quả nào."
    # Phân bố theo video giúp model biết kết quả đã hội tụ hay còn tản mát
    dist: dict[str, int] = {}
    for r in results:
        video = r.get("video_name") or r.get("video_id") or "?"
        dist[video] = dist.get(video, 0) + 1
    top_videos = sorted(dist.items(), key=lambda kv: -kv[1])[:6]
    dist_txt = ", ".join(f"{v} ({n})" for v, n in top_videos)

    lines = []
    for i, r in enumerate(results[:MAX_RESULTS_TO_MODEL], 1):
        ts = r.get("timestamp_sec")
        cap = (r.get("caption") or "")[:90]
        text = (r.get("text") or "")[:80]
        lines.append(
            f"{i}. {r.get('video_name') or r.get('video_id') or '?'} @ {ts}s"
            f" | score {r.get('score', 0):.3f}"
            + (f" | caption: {cap}…" if cap else "")
            + (f" | text: {text}" if text else "")
        )
    return (
        f"{len(results)} kết quả, trải trên {len(dist)} video: {dist_txt}\n"
        f"Top {min(len(results), MAX_RESULTS_TO_MODEL)}:\n" + "\n".join(lines)
    )


def _parse_tool_json(text: str) -> dict | None:
    """Extract the agent's single JSON object from the LLM output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _turn(
    message: str,
    results: list[dict],
    actions: list[dict],
    done: bool,
    question: str = "",
    suggestions: list[str] | None = None,
) -> dict:
    return {
        "message": message,
        "results": results,
        "actions": actions,
        "question": question,
        "suggestions": suggestions or [],
        "done": done,
    }


_AGENTS: dict[str, InteractiveAgent] = {}


def run_agent_turn(messages: list[dict], experiment: Experiment) -> dict:
    """Module-level entry point; caches one agent (and its retriever) per experiment."""
    agent = _AGENTS.get(experiment.name)
    if agent is None:
        agent = InteractiveAgent(experiment)
        _AGENTS[experiment.name] = agent
    return agent.run_turn(messages)
