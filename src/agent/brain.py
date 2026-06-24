"""VLM Brain - Gemini API wrapper for ReAct reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass
class BrainResponse:
    """Structured response from the VLM brain."""

    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    answer: str = ""
    finished: bool = False


SYSTEM_PROMPT = """Bạn là agent VQA (Video Question Answering). Nhiệm vụ: trả lời câu hỏi về khung hình video.

## Tools (Công cụ)
Bạn có các công cụ sau:
1. **caption(image_path)**: Mô tả hình ảnh tổng quan, hành động, màu sắc, con người, đồ vật, phong cảnh.
2. **ocr(image_path)**: Đọc và trích xuất chính xác văn bản, chữ viết, tên riêng, câu thơ, số trên biển hiệu/băng rôn.

## Luật bắt buộc
1. Bắt buộc suy nghĩ (thought) kỹ câu hỏi để chọn đúng tool:
   - Nếu câu hỏi tìm TÊN XÃ, CHỮ VIẾT, SỐ, CÂU THƠ, BIỂN BÁO -> BẮT BUỘC gọi tool `ocr`.
   - Nếu câu hỏi hỏi về HÀNH ĐỘNG, MÀU SẮC, SỐ LƯỢNG NGƯỜI/VẬT -> BẮT BUỘC gọi tool `caption`.
2. Trả lời bằng NGÔN NGỮ của câu hỏi.
3. Chỉ gọi tool 1 lần, đối chiếu kết quả trả về với câu hỏi để đưa ra đáp án CHÍNH XÁC.
4. Đáp án ngắn gọn, đi thẳng vào vấn đề. Nếu kết quả tool không chứa thông tin, hãy trả lời "Không tìm thấy thông tin trong khung hình".

## Định dạng JSON (Luôn trả về JSON hợp lệ)

- Khi cần gọi tool:
{{"thought": "Câu hỏi hỏi về tên xã, mình cần đọc chữ trong ảnh để tìm tên", "action": "ocr", "action_input": {{"image_path": "file.jpg"}}}}

- Khi đã có thông tin để trả lời:
{{"thought": "Kết quả OCR trả về có nhắc đến 'Xã Diên Điền', vậy đây là đáp án", "answer": "Xã Diên Điền", "finished": true}}
"""


class VlmBrain:
    """VLM brain that powers the Agent's reasoning via Gemini API."""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite") -> None:
        self.model_name = model_name
        self._client = None
        self._messages: list[dict] = []

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

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []

    def reason(
        self,
        question: str,
        shot_info: str,
        frame_count: int,
        tool_results: list[dict] | None = None,
    ) -> BrainResponse:
        """Send context + question to the VLM and parse structured response.

        Args:
            question: The VQA question to answer.
            shot_info: Description of the shot (video info, timestamps).
            frame_count: Number of frames in the shot.
            tool_results: Previous tool call results (for ReAct loop).

        Returns:
            BrainResponse with thought, action, answer.
        """
        if not self._init_client():
            return BrainResponse(answer="VLM unavailable: set GEMINI_API_KEY", finished=True)

        # Build conversation
        messages = [{"role": "user", "parts": [SYSTEM_PROMPT]}]

        context = (
            f"Question: {question}\n"
            f"Shot info: {shot_info}\n"
            f"Frames available: {frame_count}\n"
        )
        messages.append({"role": "user", "parts": [context]})

        if tool_results:
            history = "\n".join(
                f"Tool {r.get('tool', '?')} returned: {r.get('result', '')}" for r in tool_results
            )
            messages.append({"role": "user", "parts": [f"Previous observations:\n{history}"]})

        try:
            from google.genai import types

            full_prompt = messages[-1]["parts"][0]
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )

            return self._parse_response(response.text.strip())

        except Exception as exc:
            LOGGER.exception("Brain reasoning failed: %s", exc)
            return BrainResponse(answer=f"Reasoning error: {exc}", finished=True)

    def _parse_response(self, text: str) -> BrainResponse:
        """Parse JSON response from the VLM."""
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            LOGGER.warning("No JSON found in brain response: %s", text[:200])
            return BrainResponse(answer=text.strip(), finished=True)

        try:
            data = json.loads(json_match.group())
            if data.get("finished") or data.get("answer"):
                return BrainResponse(
                    thought=data.get("thought", ""),
                    answer=data.get("answer", ""),
                    finished=True,
                )
            return BrainResponse(
                thought=data.get("thought", ""),
                action=data.get("action", ""),
                action_input=data.get("action_input", {}),
                finished=False,
            )
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse brain JSON: %s", exc)
            return BrainResponse(answer=text.strip(), finished=True)

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
