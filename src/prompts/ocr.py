"""OCR prompt for on-screen text extraction from Vietnamese news keyframes.

Separate from ``prompts/captioning.py`` on purpose: captioning describes the
whole scene (for the Vietnamese embedding branch), while this prompt does
ONE thing — transcribe on-screen text verbatim — for the Elasticsearch BM25
branch (``stores/text``). Mixing the two into one call would force a
trade-off between caption fluency and OCR completeness; keeping them
separate lets each be tuned independently and keeps the Elasticsearch
document a clean transcript instead of a caption paragraph with quoted
fragments buried in it.

English instructions (fewer input tokens); the model still transcribes
on-screen text verbatim in its original language, so Vietnamese keyframes
still produce Vietnamese output.
"""

from __future__ import annotations

NO_TEXT_MARKER = "NO_TEXT"

OCR_SYSTEM_PROMPT = (
    "Extract ALL text that is actually visible ON SCREEN in this image, exactly as written "
    "(verbatim, keep the original language and diacritics — e.g. Vietnamese). Include names, "
    "place names, headlines, captions, tickers, signs, labels, on-screen graphics. "
    "Output ONLY the text, each distinct piece on its own line, no explanations, no translation. "
    f"If there is NO readable text in the image, output exactly: {NO_TEXT_MARKER}"
)

OCR_USER_PROMPT = "Extract the on-screen text from this image, one piece per line."


def build_ocr_prompt() -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for VLM on-screen text extraction."""
    return OCR_SYSTEM_PROMPT, OCR_USER_PROMPT
