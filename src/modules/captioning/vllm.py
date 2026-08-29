"""VLM captioning backed by a self-hosted vLLM OpenAI-compatible server.

Calls the chat-completions endpoint with the keyframe image inline (base64
data URL) plus the structured Vietnamese news-captioning prompt from
``prompts/captioning.py``. See that module's docstring for why the prompt is
rigid rather than free-form.
"""

from __future__ import annotations

import logging

from core.errors import CaptioningError
from modules._vllm_chat import VllmChatClient
from modules.captioning.base import CaptioningModel
from modules.captioning.validation import validate_caption
from prompts.captioning import build_caption_prompt, few_shot_messages

LOGGER = logging.getLogger(__name__)

# temperature=0 (greedy decoding) + fixed seed for maximum consistency across
# hundreds of thousands of calls — this is a batch indexing job feeding an
# embedding model, not a creative-writing task, so we optimize for zero
# lexical variance over near-duplicate keyframes rather than fluency. See
# prompts/captioning.py.
# Headroom, not a target: sampled news keyframes finish at ~150 tokens (max
# 162 measured), all with finish_reason=stop, so the cap never truncates a
# caption even on unusually text-dense frames.
_GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.1,
    "max_tokens": 384,
    "seed": 42,
}

_SEMANTIC_RETRIES = 1
_CORRECTIVE_PROMPT = (
    "Output trước không hợp lệ. Hãy tạo lại mô tả hoàn toàn bằng tiếng Việt, "
    "không chứa chữ Hán/CJK, không lặp token, không giải thích."
)
_SANITIZE_SYSTEM_PROMPT = """Bạn là bộ lọc làm sạch caption tiếng Việt cho hệ thống tìm kiếm video.
Nhiệm vụ của bạn là viết lại caption đầu vào thành MỘT đoạn tiếng Việt thuần.

QUY TẮC BẮT BUỘC:
- Không chứa bất kỳ chữ Hán/CJK nào.
- Không chép lại nguyên văn chữ Trung Quốc, Nhật, Hàn; nếu cần chỉ viết bằng tiếng Việt
  rằng trên màn hình có chữ Trung Quốc, chữ Nhật hoặc chữ Hàn.
- Giữ lại ý nghĩa mô tả cảnh, đối tượng, hành động và chữ Latin/số nếu có.
- Không thêm giải thích, không tự nhận lỗi, không viết markdown.
- Không lặp từ/cụm từ bất thường.
- Độ dài khoảng 50-100 từ."""
_SANITIZE_GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.1,
    "max_tokens": 384,
    "seed": 42,
}


class VllmCaptioningModel(CaptioningModel):
    """Caption keyframes by calling OpenRouter's VLM chat-completions endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = VllmChatClient(openrouter_model=model_name, timeout=timeout)

    def caption(self, frame_path: str) -> str:
        """Return a structured Vietnamese caption for one keyframe image."""
        system_prompt, user_prompt = build_caption_prompt()
        try:
            last_reasons: tuple[str, ...] = ()
            last_caption = ""
            for attempt in range(_SEMANTIC_RETRIES + 1):
                prompt = user_prompt if attempt == 0 else f"{_CORRECTIVE_PROMPT}\n\n{user_prompt}"
                caption = self._client.complete_with_image(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    image_path=frame_path,
                    generation_params=_GENERATION_PARAMS,
                    extra_messages=few_shot_messages(),
                )
                validation = validate_caption(caption)
                if validation.valid:
                    return caption
                last_caption = caption
                last_reasons = validation.reasons
                LOGGER.warning(
                    "Invalid caption for %s (attempt %s/%s): %s",
                    frame_path,
                    attempt + 1,
                    _SEMANTIC_RETRIES + 1,
                    ", ".join(last_reasons),
                )
            sanitized = self._sanitize_caption(last_caption, frame_path, last_reasons)
            if sanitized is not None:
                return sanitized
            raise CaptioningError(
                f"Invalid caption after semantic retry/sanitize for {frame_path}: "
                f"{', '.join(last_reasons)}"
            )
        except Exception as exc:
            LOGGER.exception("vLLM captioning failed for %s: %s", frame_path, exc)
            raise CaptioningError(f"vLLM captioning failed for {frame_path}: {exc}") from exc

    def _sanitize_caption(
        self, caption: str, frame_path: str, reasons: tuple[str, ...]
    ) -> str | None:
        if not caption.strip():
            return None
        user_prompt = (
            "Caption đầu vào không hợp lệ vì: "
            f"{', '.join(reasons)}.\n\n"
            "Hãy viết lại caption này hoàn toàn bằng tiếng Việt, loại bỏ mọi chữ Hán/CJK "
            "nhưng giữ ý nghĩa mô tả cảnh nếu có thể:\n\n"
            f"{caption}"
        )
        sanitized = self._client.complete_text(
            system_prompt=_SANITIZE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            generation_params=_SANITIZE_GENERATION_PARAMS,
        )
        validation = validate_caption(sanitized)
        if validation.valid:
            LOGGER.info("Sanitized invalid caption for %s", frame_path)
            return sanitized
        LOGGER.warning(
            "Sanitized caption still invalid for %s: %s",
            frame_path,
            ", ".join(validation.reasons),
        )
        return None
