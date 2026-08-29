"""Validation helpers for Vietnamese VLM captions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_META_PHRASES = (
    "lỗi hệ thống",
    "xu ly sai",
    "xử lý sai",
    "sửa lại",
    "sua lai",
    "đoạn trên",
    "doan tren",
    "tôi xin sửa",
    "toi xin sua",
    "xin lỗi",
    "xin loi",
    "output trước",
    "output truoc",
)
_MAX_WORDS = 180
_MAX_CHARS = 1200
_MAX_SAME_CHAR_RUN = 8
_MAX_TOKEN_REPEAT = 6


@dataclass(frozen=True)
class CaptionValidationResult:
    valid: bool
    reasons: tuple[str, ...] = ()


def validate_caption(caption: str | None) -> CaptionValidationResult:
    """Return whether ``caption`` is safe to feed into Vietnamese embedding."""
    reasons = tuple(_invalid_reasons(caption))
    return CaptionValidationResult(valid=not reasons, reasons=reasons)


def _invalid_reasons(caption: str | None) -> Iterable[str]:
    if caption is None or not caption.strip():
        yield "empty"
        return

    text = caption.strip()
    lowered = text.lower()
    words = _WORD_RE.findall(text)

    if _HAN_RE.search(text):
        yield "contains_han_cjk"

    if len(text) > _MAX_CHARS:
        yield "too_long_chars"

    if len(words) > _MAX_WORDS:
        yield "too_long_words"

    for phrase in _META_PHRASES:
        if phrase in lowered:
            yield "meta_or_error_phrase"
            break

    if _has_repeated_character_run(text):
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
            if all(
                lowered[start + size * rep : start + size * (rep + 1)] == ngram
                for rep in range(1, 4)
            ):
                return True
    return False
