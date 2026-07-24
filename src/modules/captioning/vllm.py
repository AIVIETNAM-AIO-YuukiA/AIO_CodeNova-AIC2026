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
from prompts.captioning import build_caption_prompt, few_shot_messages

LOGGER = logging.getLogger(__name__)

# temperature=0 (greedy decoding) + fixed seed for maximum consistency across
# hundreds of thousands of calls — this is a batch indexing job feeding an
# embedding model, not a creative-writing task, so we optimize for zero
# lexical variance over near-duplicate keyframes rather than fluency. See
# prompts/captioning.py.
_GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "max_tokens": 220,
    "seed": 42,
}


class VllmCaptioningModel(CaptioningModel):
    """Caption keyframes by calling a self-hosted vLLM chat-completions endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = VllmChatClient(base_url=base_url, model_name=model_name, timeout=timeout)

    def caption(self, frame_path: str) -> str:
        """Return a structured Vietnamese caption for one keyframe image."""
        system_prompt, user_prompt = build_caption_prompt()
        try:
            return self._client.complete_with_image(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=frame_path,
                generation_params=_GENERATION_PARAMS,
                extra_messages=few_shot_messages(),
            )
        except Exception as exc:
            LOGGER.exception("vLLM captioning failed for %s: %s", frame_path, exc)
            raise CaptioningError(f"vLLM captioning failed for {frame_path}: {exc}") from exc
