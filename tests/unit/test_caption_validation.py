from modules.captioning.validation import validate_caption


def assert_invalid(caption: str, reason: str) -> None:
    result = validate_caption(caption)
    assert not result.valid
    assert reason in result.reasons


def test_valid_caption_plain_vietnamese() -> None:
    caption = (
        "Cảnh phóng sự hiện trường, ngoài trời trên đường phố, trời có nắng. "
        "Một người đàn ông mặc áo trắng đứng bên cạnh xe máy màu đen. "
        "Người đàn ông đang nói chuyện với nhóm người ở bên phải."
    )
    assert validate_caption(caption).valid


def test_valid_caption_with_numbers_and_names() -> None:
    caption = (
        'Cảnh đồ họa số liệu, trong bản tin thời sự. Bảng hiển thị số "2026" '
        'và tên "OpenRouter" bằng chữ Latin. Người dẫn chương trình đứng ở bên trái.'
    )
    assert validate_caption(caption).valid


def test_invalid_mixed_han_character() -> None:
    assert_invalid("Hai người mặc đồ phẫu术 màu xanh trong phòng.", "contains_han_cjk")


def test_invalid_mixed_han_word() -> None:
    assert_invalid("Dòng nước浑浊 màu nâu chảy qua con đường.", "contains_han_cjk")


def test_invalid_full_chinese_sentence() -> None:
    assert_invalid("画面中央有一名男子站在道路旁。", "contains_han_cjk")


def test_invalid_repeated_cjk_phrase() -> None:
    assert_invalid("Một vật thể 一块一块一块一块 xuất hiện ở giữa ảnh.", "contains_han_cjk")


def test_invalid_repeated_character_run() -> None:
    assert_invalid("Cảnh khác với chữ 露露露露露露露露 ở giữa.", "repeated_character_run")


def test_invalid_too_long() -> None:
    assert_invalid(" ".join(["người"] * 181), "too_long_words")


def test_invalid_empty() -> None:
    assert_invalid("   ", "empty")


def test_invalid_meta_output() -> None:
    assert_invalid("Đoạn trên có lỗi, tôi xin sửa lại mô tả.", "meta_or_error_phrase")
