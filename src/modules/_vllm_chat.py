"""Shared low-level client for OpenAI-compatible chat-completions endpoints.

Used by captioning, OCR, and the agent/query-processor LLM calls. If the
local vLLM engine is unreachable and ``OPENROUTER_API_KEY`` is set, requests
retry once against OpenRouter.
"""

from __future__ import annotations

import base64
import logging
import os

LOGGER = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8881/v1"
_DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_OPENROUTER_MODEL = "qwen/qwen2.5-vl-72b-instruct"


class VllmChatClient:
    """Thin wrapper around an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 60.0,
        openrouter_base_url: str | None = None,
        openrouter_model: str | None = None,
        openrouter_api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.model_name = model_name or os.environ.get("VLLM_MODEL", _DEFAULT_MODEL)
        self.timeout = timeout

        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        openrouter_base_url = openrouter_base_url or os.environ.get(
            "OPENROUTER_BASE_URL", _DEFAULT_OPENROUTER_BASE_URL
        )
        self.openrouter_base_url = openrouter_base_url.rstrip("/")
        self.openrouter_model = (
            openrouter_model
            or os.environ.get("OPENROUTER_MODEL")
            or _DEFAULT_OPENROUTER_MODEL
        )

        self._client = None
        self._openrouter_client = None

    def complete_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        generation_params: dict[str, object],
        extra_messages: list[dict[str, object]] | None = None,
    ) -> str:
        """Send one chat-completions call with an inline image and return the text response."""
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

        response = self._post_with_fallback(
            lambda model_name: {"model": model_name, "messages": messages, **generation_params}
        )
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_params: dict[str, object] | None = None,
        extra_messages: list[dict[str, object]] | None = None,
    ) -> str:
        """Send one text-only chat-completions call and return the text response."""
        messages: list[dict[str, object]] = [{"role": "system", "content": system_prompt}]
        if extra_messages:
            messages.extend(extra_messages)
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        response = self._post_with_fallback(
            lambda model_name: {
                "model": model_name,
                "messages": messages,
                **(generation_params or {}),
            }
        )
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def _post_with_fallback(self, build_payload):
        """POST to the local engine; on connection failure, retry once against
        OpenRouter if configured. ``build_payload(model_name)`` builds the body."""
        import httpx

        client = self._load_client()
        try:
            response = client.post("/chat/completions", json=build_payload(self.model_name))
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            if not self.openrouter_api_key:
                raise
            LOGGER.warning(
                "Local vLLM engine at %s unreachable (%s); falling back to OpenRouter",
                self.base_url,
                exc,
            )
        openrouter_client = self._load_openrouter_client()
        response = openrouter_client.post(
            "/chat/completions", json=build_payload(self.openrouter_model)
        )
        response.raise_for_status()
        return response

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

    def _load_openrouter_client(self):
        if self._openrouter_client is not None:
            return self._openrouter_client
        import httpx

        limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
        headers = {"Authorization": f"Bearer {self.openrouter_api_key}"}
        self._openrouter_client = httpx.Client(
            base_url=self.openrouter_base_url, timeout=self.timeout, limits=limits, headers=headers
        )
        return self._openrouter_client


def _encode_image_data_url(image_path: str) -> str:
    """Read an image file and return it as a base64 data URL."""
    suffix = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpg"
    mime = "image/png" if suffix == "png" else "image/jpeg"
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
