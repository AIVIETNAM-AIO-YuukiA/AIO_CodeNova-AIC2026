from modules.ocr.vllm import VllmOcrModel


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete_with_image(self, **kwargs) -> str:
        return self.response


def test_ocr_allows_no_text_marker() -> None:
    model = VllmOcrModel()
    model._client = FakeClient("NO_TEXT")

    assert model.recognize("/tmp/frame.jpg") == ""


def test_ocr_strips_think_block() -> None:
    model = VllmOcrModel()
    model._client = FakeClient("<think>reasoning...</think>VTV24\nHa Noi")

    assert model.recognize("/tmp/frame.jpg") == "VTV24\nHa Noi"


def test_ocr_allows_cjk_text() -> None:
    model = VllmOcrModel()
    model._client = FakeClient("共同发展\n東京都")

    assert model.recognize("/tmp/frame.jpg") == "共同发展\n東京都"


def test_ocr_collapses_repeated_lines_instead_of_failing() -> None:
    # Observed in production: the model gets stuck re-emitting the same
    # banner line dozens of times on frames with many repeated banners.
    model = VllmOcrModel()
    model._client = FakeClient("HTV\n" + "TON DONG A\n" * 20)

    assert model.recognize("/tmp/frame.jpg") == "HTV\nTON DONG A"
