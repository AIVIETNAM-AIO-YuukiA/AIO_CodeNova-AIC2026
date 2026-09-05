"""On-screen text extraction over the shared VLM client (see modules/_vllm_chat.py)."""

from __future__ import annotations

import logging

from core.errors import CaptioningError
from modules._vllm_chat import VllmChatClient
from modules.ocr.base import OcrModel
from modules.ocr.validation import validate_ocr_text
from prompts.ocr import NO_TEXT_MARKER, build_ocr_prompt

LOGGER = logging.getLogger(__name__)

# Greedy decoding: transcription must be deterministic, not creative.
# Measured over sampled news keyframes: the busiest (full ticker + station
# logo + clock + captions) needed 80 tokens, so 200 leaves plenty of headroom.
_GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.1,
    "max_tokens": 200,
    "seed": 42,
}

# Repetition reasons are handled by _dedupe_consecutive_lines above; if they
# still fire afterwards (e.g. repetition isn't line-aligned), the text is kept
# rather than discarding the whole frame — other reasons (too-long, meta/error
# phrases) still indicate a genuinely bad response and are rejected.
_REPETITION_REASONS = frozenset(
    {"repeated_character_run", "repeated_token_run", "repeated_ngram_run"}
)


class VllmOcrModel(OcrModel):
    """Extract on-screen text from keyframes via OpenRouter's VLM endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = VllmChatClient(openrouter_model=model_name, timeout=timeout)

    def recognize(self, frame_path: str) -> str:
        """Return on-screen text for one frame image, or "" if none is visible."""
        system_prompt, user_prompt = build_ocr_prompt()
        try:
            text = self._client.complete_with_image(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=frame_path,
                generation_params=_GENERATION_PARAMS,
            )
        except Exception as exc:
            LOGGER.exception("vLLM OCR failed for %s: %s", frame_path, exc)
            raise CaptioningError(f"vLLM OCR failed for {frame_path}: {exc}") from exc

        text = text.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        text = "" if text == NO_TEXT_MARKER else text
        # The model occasionally gets stuck re-emitting the same line dozens of
        # times (observed on frames with many repeated banners along a shot's
        # background) instead of a network/decoding failure — collapse that
        # in place rather than discarding the whole frame, same as how AIC
        # 2025's OCR script accepts every response and only prunes noise
        # afterwards (its cross-document watermark filter, mirrored in
        # extract_text._drop_watermark_lines) rather than rejecting responses.
        text = _dedupe_consecutive_lines(text)

        validation = validate_ocr_text(text)
        if not validation.valid:
            reasons = ", ".join(validation.reasons)
            if set(validation.reasons) <= _REPETITION_REASONS:
                LOGGER.warning(
                    "OCR output for %s still repetitive after line-dedupe (%s); keeping as-is",
                    frame_path,
                    reasons,
                )
            else:
                LOGGER.warning("Invalid OCR output for %s: %s", frame_path, reasons)
                raise CaptioningError(f"Invalid OCR output for {frame_path}: {reasons}")
        return text


def _dedupe_consecutive_lines(text: str) -> str:
    """Collapse runs of identical consecutive lines to a single occurrence."""
    lines = text.splitlines()
    deduped: list[str] = []
    for line in lines:
        if not deduped or line.strip() != deduped[-1].strip():
            deduped.append(line)
    return "\n".join(deduped)
