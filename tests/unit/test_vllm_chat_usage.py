import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import httpx

from modules._vllm_chat import VllmChatClient


class _Response:
    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.001},
        }


def test_text_completion_exposes_backend_usage(monkeypatch) -> None:
    client = VllmChatClient(max_retries=0)
    monkeypatch.setattr(client, "_post_openrouter", lambda build_payload: _Response())

    assert client.complete_text("system", "user") == "ok"
    assert client.last_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "cost": 0.001,
    }


def test_query_router_can_disable_shared_internal_retries() -> None:
    client = VllmChatClient(max_retries=0)
    assert client.max_retries == 0


def test_usage_is_isolated_between_concurrent_request_threads(monkeypatch) -> None:
    client = VllmChatClient(max_retries=0)
    barrier = threading.Barrier(2)

    class ThreadResponse:
        def __init__(self, request_id: str) -> None:
            self.request_id = request_id

        def json(self):
            return {
                "choices": [{"message": {"content": self.request_id}}],
                "usage": {"request_id": self.request_id},
            }

    monkeypatch.setattr(
        client,
        "_post_openrouter",
        lambda build_payload: ThreadResponse(threading.current_thread().name),
    )

    def run_request() -> tuple[str, str]:
        answer = client.complete_text("system", "user")
        barrier.wait()
        return answer, str(client.last_usage["request_id"])

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="request") as pool:
        results = list(pool.map(lambda _: run_request(), range(2)))

    assert all(answer == usage_id for answer, usage_id in results)
    assert len({answer for answer, _ in results}) == 2


def test_real_http_attempt_count_includes_retry(monkeypatch) -> None:
    client = VllmChatClient(
        openrouter_api_key="test-key",
        openrouter_model="vision-model",
        max_retries=1,
    )
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    responses = iter(
        [
            httpx.Response(503, request=request),
            httpx.Response(
                200,
                request=request,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            ),
        ]
    )
    transport = MagicMock()
    transport.post.side_effect = lambda *args, **kwargs: next(responses)
    monkeypatch.setattr(client, "_load_openrouter_client", lambda: transport)
    monkeypatch.setattr("modules._vllm_chat._sleep_before_retry", lambda *args: None)

    assert client.complete_text("system", "user") == "ok"
    assert client.last_usage["request_count"] == 2
    assert transport.post.call_count == 2
