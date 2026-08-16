"""Validation helpers for OCR VLM output."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MAX_CHARS = 1200
_MAX_LINES = 40
_MAX_WORDS = 220
_MAX_SAME_CHAR_RUN = 8
_MAX_TOKEN_REPEAT = 8
_META_PHRASES = (
    "xin lỗi",
    "xin loi",
    "tôi không thể",
    "toi khong the",
    "tôi không thể hỗ trợ",
    "không thể xử lý",
    "khong the xu ly",
    "lỗi hệ thống",
    "loi he thong",
    "xử lý sai",
    "xu ly sai",
    "sửa lại",
    "sua lai",
    "đoạn trên",
    "doan tren",
    "output trước",
    "output truoc",
    "as an ai",
    "i'm sorry",
    "i am sorry",
)
_SCENE_DESCRIPTION_STARTS = (
    "đây là cảnh",
    "day la canh",
    "cảnh phóng sự",
    "canh phong su",
    "cảnh trường quay",
    "canh truong quay",
    "trong ảnh",
    "trong anh",
    "hình ảnh cho thấy",
    "hinh anh cho thay",
    "khung hình",
    "khung hinh",
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
    non_empty_lines = [line for line in stripped.splitlines() if line.strip()]

    if len(stripped) > _MAX_CHARS:
        yield "too_long_chars"

    if len(words) > _MAX_WORDS:
        yield "too_long_words"

    if len(non_empty_lines) > _MAX_LINES:
        yield "too_many_lines"

    for phrase in _META_PHRASES:
        if phrase in lowered:
            yield "meta_or_error_phrase"
            break

    if any(lowered.startswith(prefix) for prefix in _SCENE_DESCRIPTION_STARTS):
        yield "scene_description"

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
