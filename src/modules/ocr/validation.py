"""Validation helpers for OCR VLM output."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MAX_SAME_CHAR_RUN = 8
_MAX_TOKEN_REPEAT = 8
_META_PHRASES = (
    "as an ai",
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
)
# Word-boundary matching: a plain substring check flags real on-screen text
# like an "AI Cannot" slide heading, whose lowercase form "ai cannot" contains
# "i cannot" as a substring.
_META_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(phrase) for phrase in _META_PHRASES) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OcrValidationResult:
    valid: bool
    reasons: tuple[str, ...] = ()


def validate_ocr_text(text: str | None) -> OcrValidationResult:
    """Return whether OCR output is safe to store as searchable text.

    Empty output is valid: many keyframes have no visible text. Unlike caption
    validation, this deliberately allows Han/CJK and other non-Vietnamese text
    because OCR should transcribe whatever appears on screen.
    """
    reasons = tuple(_invalid_reasons(text))
    return OcrValidationResult(valid=not reasons, reasons=reasons)


def _invalid_reasons(text: str | None) -> Iterable[str]:
    if text is None or not text.strip():
        return

    stripped = text.strip()
    lowered = stripped.lower()
    words = _WORD_RE.findall(stripped)

    if _META_PHRASE_RE.search(lowered):
        yield "meta_or_error_phrase"

    if _has_repeated_character_run(stripped):
        yield "repeated_character_run"

    if _has_repeated_token_run(words):
        yield "repeated_token_run"

    if _has_repeated_ngram_run(words):
        yield "repeated_ngram_run"


def _has_repeated_character_run(text: str) -> bool:
    previous = ""
    run = 0
    for char in text:
        if char.isspace():
            previous = ""
            run = 0
            continue
        if char == previous:
            run += 1
        else:
            previous = char
            run = 1
        if run >= _MAX_SAME_CHAR_RUN:
            return True
    return False


def _has_repeated_token_run(words: list[str]) -> bool:
    previous = ""
    run = 0
    for word in (word.lower() for word in words):
        if word == previous:
            run += 1
        else:
            previous = word
            run = 1
        if run >= _MAX_TOKEN_REPEAT:
            return True
    return False


def _has_repeated_ngram_run(words: list[str]) -> bool:
    lowered = [word.lower() for word in words]
    for size in (2, 3, 4):
        if len(lowered) < size * 4:
            continue
        for start in range(0, len(lowered) - size * 4 + 1):
            ngram = lowered[start : start + size]
            repeated = True
            for rep in range(1, 4):
                begin = start + size * rep
                if lowered[begin : begin + size] != ngram:
                    repeated = False
                    break
            if repeated:
                return True
    return False
