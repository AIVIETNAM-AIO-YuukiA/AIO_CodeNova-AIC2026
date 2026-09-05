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


def test_invalid_meta_output() -> None:
    assert_invalid("I'm sorry, I cannot process this image.", "meta_or_error_phrase")


def test_meta_phrase_does_not_false_positive_on_real_text() -> None:
    # "AI Cannot" is a real on-screen slide heading — its lowercase form
    # "ai cannot" must not be flagged just because it contains "i cannot".
    text = "AI Cannot\n- Perform commonsense reasoning\n- Create a unique thought"
    assert validate_ocr_text(text).valid


def test_dense_but_real_text_is_valid() -> None:
    # Screens with many real UI labels (e.g. video editor software) can have
    # far more lines/words than a normal caption without being garbage.
    text = "\n".join(f"Label {i}" for i in range(60))
    assert validate_ocr_text(text).valid
