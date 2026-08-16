from modules.ocr.validation import validate_ocr_text


def assert_invalid(text: str, reason: str) -> None:
    result = validate_ocr_text(text)
    assert not result.valid
    assert reason in result.reasons


def test_empty_ocr_is_valid() -> None:
    assert validate_ocr_text("").valid
    assert validate_ocr_text("   ").valid


def test_regular_ocr_lines_are_valid() -> None:
    text = "THỜI SỰ 18H30\nMiền Trung khắc phục hậu quả sau bão số 6\n2026"
    assert validate_ocr_text(text).valid


def test_cjk_ocr_is_valid() -> None:
    assert validate_ocr_text("共同发展\n東京都\n뉴스").valid


def test_invalid_repeated_character_run() -> None:
    assert_invalid("AAAAAAAA", "repeated_character_run")


def test_invalid_repeated_token_run() -> None:
    assert_invalid("test test test test test test test test", "repeated_token_run")


def test_invalid_repeated_ngram_run() -> None:
    assert_invalid("abc def abc def abc def abc def", "repeated_ngram_run")


def test_invalid_too_long() -> None:
    assert_invalid(" ".join(["text"] * 221), "too_long_words")


def test_invalid_meta_output() -> None:
    assert_invalid("Xin lỗi, tôi không thể xử lý ảnh này.", "meta_or_error_phrase")


def test_invalid_scene_description() -> None:
    assert_invalid("Đây là cảnh phóng sự hiện trường với một người đàn ông.", "scene_description")
