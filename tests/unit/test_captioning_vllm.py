from modules.captioning.vllm import VllmCaptioningModel


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.text_responses = []
        self.calls = []
        self.text_calls = []

    def complete_with_image(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def complete_text(self, **kwargs) -> str:
        self.text_calls.append(kwargs)
        return self.text_responses.pop(0)


def test_caption_retries_invalid_output() -> None:
    model = VllmCaptioningModel()
    fake = FakeClient(
        [
            "Hai người mặc đồ phẫu术 màu xanh.",
            "Cảnh phóng sự hiện trường, trong nhà. Một nhóm người mặc áo xanh đứng quanh bàn. Họ đang trao đổi và nhìn vào tài liệu ở giữa bàn.",
        ]
    )
    model._client = fake

    caption = model.caption("/tmp/frame.jpg")

    assert "phẫu" not in caption
    assert len(fake.calls) == 2
    assert "Output trước không hợp lệ" in fake.calls[1]["user_prompt"]


def test_caption_sanitizes_after_image_retries_fail() -> None:
    model = VllmCaptioningModel()
    fake = FakeClient(
        [
            "Cảnh phóng sự có dòng chữ 無洗米 trên màn hình.",
            "Cảnh phóng sự có dòng chữ 無洗米 trên màn hình.",
        ]
    )
    fake.text_responses = [
        "Cảnh phóng sự hiện trường trong một cửa hàng, trên màn hình có chữ Nhật liên quan đến sản phẩm gạo. Một người đàn ông đứng gần các bao hàng và bảng thông tin. Khung hình có logo kênh truyền hình và thanh tin tức ở phía dưới."
    ]
    model._client = fake

    caption = model.caption("/tmp/frame.jpg")

    assert "無" not in caption
    assert "chữ Nhật" in caption
    assert len(fake.calls) == 2
    assert len(fake.text_calls) == 1
