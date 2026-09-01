from __future__ import annotations

from pathlib import Path

import pytest

from modules._vllm_chat import VllmChatClient


class _Response:
    def json(self) -> dict[str, object]:
        return {
            "choices": [{"message": {"content": " grounded answer "}}],
            "usage": {"prompt_tokens": 41, "completion_tokens": 7, "cost": 0.002},
        }


def _write_image(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def test_multi_image_completion_preserves_labels_order_detail_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _write_image(tmp_path / "first.jpg", b"first-frame")
    second = _write_image(tmp_path / "second.png", b"second-frame")
    captured: dict[str, object] = {}

    client = VllmChatClient(max_retries=0)

    def capture_payload(build_payload):
        captured.update(build_payload("vision-model"))
        return _Response()

    monkeypatch.setattr(client, "_post_openrouter", capture_payload)

    answer = client.complete_with_images(
        system_prompt="ground answers in cited frames",
        user_prompt="What is X?",
        image_paths=[first, second],
        image_labels=["F1 @ 12.3s", "F2 @ 15.8s"],
        detail="high",
        generation_params={"temperature": 0, "max_tokens": 128},
    )

    assert answer == "grounded answer"
    assert client.last_usage == {
        "prompt_tokens": 41,
        "completion_tokens": 7,
        "cost": 0.002,
    }
    assert captured["model"] == "vision-model"
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 128

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "system", "content": "ground answers in cited frames"}
    content = messages[-1]["content"]
    assert [block["type"] for block in content] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    assert content[0]["text"] == "What is X?"
    assert content[1]["text"] == "[F1 @ 12.3s]"
    assert content[3]["text"] == "[F2 @ 15.8s]"
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[4]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"]["detail"] == "high"
    assert content[4]["image_url"]["detail"] == "high"


def test_multi_image_completion_uses_default_labels_and_can_omit_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _write_image(tmp_path / "first.jpg", b"first-frame")
    second = _write_image(tmp_path / "second.jpg", b"second-frame")
    captured: dict[str, object] = {}
    client = VllmChatClient(max_retries=0)

    def capture_payload(build_payload):
        captured.update(build_payload("vision-model"))
        return _Response()

    monkeypatch.setattr(client, "_post_openrouter", capture_payload)
    client.complete_with_images("system", "question", [first, second], detail=None)

    content = captured["messages"][-1]["content"]
    assert content[1] == {"type": "text", "text": "[F1]"}
    assert content[3] == {"type": "text", "text": "[F2]"}
    assert "detail" not in content[2]["image_url"]
    assert "detail" not in content[4]["image_url"]


def test_multi_image_completion_rejects_empty_or_more_than_six_images() -> None:
    client = VllmChatClient(max_retries=0)

    with pytest.raises(ValueError, match="at least one"):
        client.complete_with_images("system", "question", [])
    with pytest.raises(ValueError, match="at most 6"):
        client.complete_with_images("system", "question", ["frame.jpg"] * 7)


def test_multi_image_completion_validates_all_files_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _write_image(tmp_path / "existing.jpg", b"frame")
    missing = tmp_path / "missing.jpg"
    client = VllmChatClient(max_retries=0)

    def unexpected_request(build_payload):
        raise AssertionError("OpenRouter must not be called when an image is missing")

    monkeypatch.setattr(client, "_post_openrouter", unexpected_request)

    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        client.complete_with_images("system", "question", [existing, missing])


def test_multi_image_completion_validates_labels_and_detail(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "frame.jpg", b"frame")
    client = VllmChatClient(max_retries=0)

    with pytest.raises(ValueError, match="exactly one label"):
        client.complete_with_images("system", "question", [image], image_labels=[])
    with pytest.raises(ValueError, match="must not be empty"):
        client.complete_with_images("system", "question", [image], image_labels=["  "])
    with pytest.raises(ValueError, match="detail must be one of"):
        client.complete_with_images("system", "question", [image], detail="original")
