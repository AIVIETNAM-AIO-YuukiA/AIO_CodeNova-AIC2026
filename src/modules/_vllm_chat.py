"""Shared low-level client for calling a self-hosted vLLM chat-completions endpoint.

Internal helper — not a public module interface. Used by both
``modules/captioning/vllm.py`` (scene captioning) and ``modules/ocr/vllm.py``
(on-screen text extraction): same server, same model, same image-inlining
mechanics, different prompts and different downstream consumers (Vietnamese
embedding branch vs. Elasticsearch branch).
"""

from __future__ import annotations

import base64
import os

_DEFAULT_BASE_URL = "http://localhost:8881/v1"
_DEFAULT_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


class VllmChatClient:
    """Thin wrapper around vLLM's OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.model_name = model_name or os.environ.get("VLLM_MODEL", _DEFAULT_MODEL)
        self.timeout = timeout
        self._client = None

    def complete_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        generation_params: dict[str, object],
        extra_messages: list[dict[str, object]] | None = None,
    ) -> str:
        """Send one chat-completions call with an inline image and return the text response."""
        client = self._load_client()
        image_data_url = _encode_image_data_url(image_path)

        messages: list[dict[str, object]] = [{"role": "system", "content": system_prompt}]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        )

        response = client.post(
            "/chat/completions",
            json={"model": self.model_name, "messages": messages, **generation_params},
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def _load_client(self):
        # httpx.Client is thread-safe (each request gets a connection from the
        # shared pool), so a single instance can be called concurrently from
        # the ThreadPoolExecutor in VietnameseEmbedder/VllmOcrModel.
        if self._client is not None:
            return self._client
        import httpx

        limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, limits=limits)
        return self._client


def _encode_image_data_url(image_path: str) -> str:
    """Read an image file and return it as a base64 data URL."""
    suffix = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpg"
    mime = "image/png" if suffix == "png" else "image/jpeg"
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
