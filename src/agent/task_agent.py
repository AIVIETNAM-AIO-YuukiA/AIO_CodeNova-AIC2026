"""AI Co-Pilot Agent for Search Routing, Query Optimization, and Verification."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from config.settings import Experiment, PipelineConfig
from core.logging import get_logger
from retrieval import build_retriever
from retrieval.vqa import vqa_search, trake_search
from retrieval.tracks import TrackQuery, build_retrieval_text

LOGGER = get_logger(__name__)

# System prompt for Task/Retrieval Agent
TASK_SYSTEM_PROMPT = """Bạn là AI Co-Pilot định tuyến và tối ưu hóa tìm kiếm (Retrieval Orchestration) cho hệ thống Video Retrieval.

Nhiệm vụ của bạn là nhận yêu cầu tìm kiếm tự nhiên của người dùng, phân tích để tự chọn hành động (Tool) phù hợp nhất, tự thực thi và quan sát (observe) kết quả để điều chỉnh truy vấn khi cần thiết nhằm đạt được kết quả tốt nhất.

## Các track tìm kiếm:
1. **Textual KIS (Known-Item Search):** Tìm kiếm cảnh video bằng văn bản mô tả thông thường.
2. **TRAKE (Temporal Retrieval):** Tìm kiếm chuỗi các sự kiện diễn ra theo thứ tự thời gian (ví dụ: "sự kiện A, sau đó đến sự kiện B").
3. **VQA (Video Question Answering):** Trả lời các câu hỏi cụ thể về nội dung video (ví dụ: "Ai là người...", "Người đó mặc áo màu gì...").

## Luật hoạt động (ReAct Loop)
Bạn hoạt động theo mô hình: Suy nghĩ (Thought) -> Chọn hành động/Tool (Action) -> Nhận kết quả (Observation) -> Tiếp tục suy nghĩ và điều chỉnh truy vấn nếu cần -> Hoàn thành.

## Cơ chế tự phục hồi & tối ưu hóa (Query Refinement):
- Khi gọi các tool search, bạn sẽ nhận về danh sách kết quả kèm theo điểm số (score).
- Nếu điểm số cao nhất (highest score) dưới 0.45 hoặc danh sách kết quả rỗng, hãy tự động SUY NGHĨ (Thought) để cải thiện câu truy vấn (ví dụ: dịch các từ khóa sang tiếng Anh tối ưu hơn, rút gọn câu mô tả rườm rà thành từ khóa đặc trưng) và gọi lại tool search thêm một lần nữa.

## Định dạng JSON bắt buộc
Mỗi lượt phản hồi của bạn phải là một đối tượng JSON hợp lệ duy nhất:

Để gọi một tool (hành động):
{
  "thought": "Lý do gọi tool, phân tích câu query hiện tại",
  "action": "tên_tool",
  "action_input": {
    "arg1": "value1"
  }
}

Khi đã có kết quả tối ưu nhất để trả về cho người dùng:
{
  "thought": "Đã tìm thấy kết quả phù hợp với điểm score tốt",
  "answer": "Nội dung giải thích chi tiết cho người dùng bằng tiếng Việt về những gì bạn tìm thấy (ví dụ: tìm thấy video nào, ở giây thứ mấy, độ khớp bao nhiêu)",
  "finished": true
}

## Danh sách Tools khả dụng:

1. **search_vector_kis(experiment_name: str, query: str, top_k: int = 5)**
   - Tìm kiếm video dựa trên một câu truy vấn văn bản đơn lẻ.
   - `query`: Nên là mô tả tiếng Anh tối ưu để khớp với CLIP (ví dụ: "a person riding a bicycle").

2. **search_trake(experiment_name: str, events: list[str], top_k: int = 5)**
   - Tìm kiếm chuỗi các sự kiện theo trình tự thời gian.
   - `events`: Danh sách các sự kiện mô tả bằng tiếng Anh (ví dụ: ["a person riding a motorbike", "a person falling down"]).

3. **search_vqa(experiment_name: str, query: str, question: str, context: str = "", top_k: int = 5)**
   - Tìm kiếm và trả lời câu hỏi cụ thể về cảnh video.
   - `query`: Câu mô tả cảnh chứa câu trả lời.
   - `question`: Câu hỏi cụ thể (ví dụ: "What color is the car?").
   - `context`: Ngữ cảnh bổ sung (nếu có).

Trả lời người dùng bằng tiếng Việt rõ ràng, chuyên nghiệp.
"""


class TaskAgent:
    """Core Retrieval Co-Pilot Agent using Gemini SDK."""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite", max_steps: int = 8, mock: bool = False) -> None:
        self.model_name = model_name
        self.max_steps = max_steps
        self.mock = mock
        self._client = None

    def _init_client(self) -> bool:
        if self._client is not None:
            return True
        self._load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            LOGGER.error("GEMINI_API_KEY environment variable not set.")
            return False
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            return True
        except Exception as exc:
            LOGGER.exception("Failed to init Gemini client: %s", exc)
            return False

    @staticmethod
    def _load_dotenv() -> None:
        """Load .env file if GEMINI_API_KEY is not already set."""
        if os.environ.get("GEMINI_API_KEY"):
            return
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    if key.strip() == "GEMINI_API_KEY":
                        os.environ["GEMINI_API_KEY"] = val.strip()
                        break

    def run_turn(self, message: str, history: list[dict[str, str]] = None) -> dict[str, Any]:
        """Execute one complete turn of agent reasoning and tool usage to answer a request."""
        if not self._init_client():
            return {
                "answer": "Lỗi: Không tìm thấy GEMINI_API_KEY trong file .env hoặc biến môi trường. Vui lòng cấu hình GEMINI_API_KEY trước.",
                "trace": [{"step": "error", "detail": "GEMINI_API_KEY missing"}],
                "finished": True,
            }

        history_context = []
        if history:
            for turn in history:
                role = "user" if turn.get("role") == "user" else "model"
                history_context.append(f"{role.upper()}: {turn.get('text')}")
        
        conversation_history = "\n".join(history_context)

        # ReAct loop
        tool_observations = []
        trace = []

        from google.genai import types

        for step in range(1, self.max_steps + 1):
            # Construct prompt for the current step
            step_prompt = f"Lịch sử hội thoại trước đó:\n{conversation_history}\n\nYêu cầu tìm kiếm của người dùng: {message}\n"
            
            if tool_observations:
                obs_text = "\n".join(
                    f"Bước {o['step']} - Tool '{o['tool']}' với input '{o['input']}' trả về kết quả:\n{o['result']}"
                    for o in tool_observations
                )
                step_prompt += f"\nKết quả thực thi các tool trước đó:\n{obs_text}\n"
            
            step_prompt += "\nHãy suy nghĩ bước tiếp theo. Chọn tool phù hợp hoặc trả về kết quả cuối cùng. Nhớ chỉ trả về 1 block JSON duy nhất."

            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        {"role": "user", "parts": [TASK_SYSTEM_PROMPT]},
                        {"role": "user", "parts": [step_prompt]}
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                
                response_text = response.text.strip()
                parsed = self._parse_json(response_text)
                
                thought = parsed.get("thought", "")
                trace.append({"step": f"thought_step_{step}", "detail": thought})
                
                # Check if finished
                if parsed.get("finished"):
                    answer = parsed.get("answer", "Đã hoàn thành tìm kiếm.")
                    return {
                        "answer": answer,
                        "trace": trace,
                        "finished": True,
                    }
                
                # Execute action
                action = parsed.get("action")
                action_input = parsed.get("action_input", {})
                
                if not action:
                    return {
                        "answer": parsed.get("answer") or "Agent dừng lại không có hành động tiếp theo.",
                        "trace": trace,
                        "finished": True,
                    }
                
                trace.append({"step": f"call_tool_{action}", "detail": f"Input: {action_input}"})
                
                # Run the actual tool
                observation = self._execute_tool(action, action_input)
                
                tool_observations.append({
                    "step": step,
                    "tool": action,
                    "input": action_input,
                    "result": observation
                })
                
                trace.append({"step": f"observation_{action}", "detail": observation[:600] + ("..." if len(observation) > 600 else "")})
                
            except Exception as exc:
                LOGGER.exception("Agent run step failed: %s", exc)
                return {
                    "answer": f"Đã xảy ra lỗi hệ thống trong quá trình Agent suy luận: {exc}",
                    "trace": trace,
                    "finished": True,
                }
        
        return {
            "answer": "Agent đã vượt quá số bước suy luận tối đa mà chưa tìm thấy kết quả phù hợp nhất.",
            "trace": trace,
            "finished": True,
        }

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Robustly parse JSON response from the LLM."""
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return {"thought": "Không thể parse JSON từ phản hồi", "answer": text, "finished": True}
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {"thought": "Lỗi decode JSON", "answer": text, "finished": True}

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> str:
        """Call corresponding python function/tool logic based on agent action."""
        try:
            # Enforce experiment name default
            experiment_name = inputs.get("experiment_name", "demo")
            
            if name == "search_vector_kis":
                return self._tool_search_vector_kis(
                    experiment_name=experiment_name,
                    query=inputs.get("query", ""),
                    top_k=inputs.get("top_k", 5)
                )
            
            elif name == "search_trake":
                return self._tool_search_trake(
                    experiment_name=experiment_name,
                    events=inputs.get("events", []),
                    top_k=inputs.get("top_k", 5)
                )
                
            elif name == "search_vqa":
                return self._tool_search_vqa(
                    experiment_name=experiment_name,
                    query=inputs.get("query", ""),
                    question=inputs.get("question", ""),
                    context=inputs.get("context", ""),
                    top_k=inputs.get("top_k", 5)
                )
            else:
                return f"Error: Tool '{name}' không tồn tại trong hệ thống."
        except Exception as e:
            return f"Error thực thi tool '{name}': {str(e)}"

    # --- Tool Implementations ---

    def _tool_search_vector_kis(self, experiment_name: str, query: str, top_k: int = 5) -> str:
        """Run KIS textual search."""
        if self.mock:
            results = []
            for i in range(1, top_k + 1):
                results.append({
                    "video_id": f"v_{i % 3 + 1:03d}",
                    "video_name": f"mock_video_{i % 3 + 1}.mp4",
                    "frame_index": i * 15,
                    "timestamp_sec": float(i * 1.5),
                    "score": 0.85 - (i * 0.03),
                    "frame_path": f"mock_frame_search_{i}.jpg"
                })
            return json.dumps({"track": "textual_kis", "top_results": results}, indent=2)
            
        try:
            config = PipelineConfig()
            experiment = Experiment.open(config=config, name=experiment_name)
            retriever = build_retriever(experiment)
            results = retriever.search(query=query, top_k=top_k)
            
            formatted = []
            for r in results:
                formatted.append({
                    "video_id": r.video_id,
                    "video_name": r.video_name,
                    "score": float(r.score),
                    "frame_index": int(r.frame_index),
                    "timestamp_sec": float(r.timestamp_sec) if r.timestamp_sec else 0.0,
                    "frame_path": str(r.frame_path) if r.frame_path else ""
                })
            return json.dumps({"track": "textual_kis", "top_results": formatted}, indent=2)
        except Exception as e:
            return f"Error search_vector_kis: {str(e)}"

    def _tool_search_trake(self, experiment_name: str, events: list[str], top_k: int = 5) -> str:
        """Run TRAKE temporal search."""
        if self.mock:
            videos = []
            for i in range(1, 4):
                video_events = []
                for idx, ev in enumerate(events):
                    video_events.append({
                        "frame_id": f"f_trake_{i}_{idx}",
                        "frame_index": idx * 120 + 20,
                        "timestamp_sec": float(idx * 6.0 + 1.2),
                        "frame_path": f"mock_trake_{i}_{idx}.jpg"
                    })
                videos.append({
                    "video_id": f"v_trake_{i:03d}",
                    "video_name": f"mock_trake_video_{i}.mp4",
                    "score": 0.88 - (i * 0.05),
                    "events": video_events
                })
            # Flatten to top_results format for UI compatibility
            flat_results = []
            for v in videos:
                for ev in v["events"]:
                    flat_results.append({
                        "video_id": v["video_id"],
                        "video_name": v["video_name"],
                        "score": v["score"],
                        "frame_index": ev["frame_index"],
                        "timestamp_sec": ev["timestamp_sec"],
                        "frame_path": ev["frame_path"]
                    })
            return json.dumps({"track": "trake", "top_results": flat_results}, indent=2)

        try:
            # Fixed top_k to 300 for TRAKE inner pipeline
            res = trake_search(
                experiment=Experiment.open(config=PipelineConfig(), name=experiment_name),
                events=events,
                top_k=300
            )
            videos = res.get("videos", [])
            flat_results = []
            for v in videos[:top_k]:
                for ev in v.get("events", []):
                    flat_results.append({
                        "video_id": v.get("video_id"),
                        "video_name": v.get("video_name"),
                        "score": float(v.get("score", 0.0)),
                        "frame_index": int(ev.get("frame_index", 0)),
                        "timestamp_sec": float(ev.get("timestamp_sec", 0.0)),
                        "frame_path": str(ev.get("frame_path", ""))
                    })
            return json.dumps({"track": "trake", "top_results": flat_results}, indent=2)
        except Exception as e:
            return f"Error search_trake: {str(e)}"

    def _tool_search_vqa(self, experiment_name: str, query: str, question: str, context: str = "", top_k: int = 5) -> str:
        """Run VQA search and answer generation."""
        if self.mock:
            results = []
            for i in range(1, 4):
                results.append({
                    "video_id": f"v_vqa_{i:03d}",
                    "video_name": f"mock_vqa_video_{i}.mp4",
                    "frame_index": 30 + i * 20,
                    "timestamp_sec": float(10.0 + i * 5.0),
                    "score": 0.92 - (i * 0.04),
                    "frame_path": f"mock_vqa_{i}.jpg"
                })
            answer = f"Đây là câu trả lời giả lập cho câu hỏi: '{question}'. Theo phân tích video mock, đối tượng xuất hiện ở giây thứ 15."
            return json.dumps({"track": "vqa", "answer": answer, "top_results": results}, indent=2)

        try:
            experiment = Experiment.open(config=PipelineConfig(), name=experiment_name)
            res = vqa_search(
                experiment=experiment,
                query=query,
                question=question,
                context=context,
                top_k=top_k
            )
            vqa_ans = res.get("answer", "Không thể trả lời.")
            results = res.get("results", [])
            formatted = []
            for r in results:
                formatted.append({
                    "video_id": r.get("video_id"),
                    "video_name": r.get("video_name"),
                    "score": float(r.get("score", 0.0)),
                    "frame_index": int(r.get("frame_index", 0)),
                    "timestamp_sec": float(r.get("timestamp_sec", 0.0)),
                    "frame_path": str(r.get("frame_path", ""))
                })
            return json.dumps({"track": "vqa", "answer": vqa_ans, "top_results": formatted}, indent=2)
        except Exception as e:
            return f"Error search_vqa: {str(e)}"
