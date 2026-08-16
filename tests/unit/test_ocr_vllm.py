import pytest

from core.errors import CaptioningError
from modules.ocr.vllm import VllmOcrModel


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete_with_image(self, **kwargs) -> str:
        return self.response


def test_ocr_allows_no_text_marker() -> None:
    model = VllmOcrModel()
    model._client = FakeClient("KHONG_CO_CHU")

    assert model.recognize("/tmp/frame.jpg") == ""


def test_ocr_rejects_scene_description() -> None:
    model = VllmOcrModel()
    model._client = FakeClient("Đây là cảnh phóng sự hiện trường có một người đàn ông.")

    with pytest.raises(CaptioningError, match="scene_description"):
        model.recognize("/tmp/frame.jpg")


def test_ocr_allows_cjk_text() -> None:
    model = VllmOcrModel()
    model._client = FakeClient("共同发展\n東京都")

    assert model.recognize("/tmp/frame.jpg") == "共同发展\n東京都"
