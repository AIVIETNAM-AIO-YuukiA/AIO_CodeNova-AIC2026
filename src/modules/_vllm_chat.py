"""Shared low-level client for OpenAI-compatible chat-completions endpoints.

Internal helper — not a public module interface. Used by
``modules/captioning/vllm.py`` / ``modules/ocr/vllm.py`` (Atlas
``atlas-index`` service, port 8881, image-inlined calls — GB10 only, no
non-GB10 fallback) and by the agent + query processor (Atlas ``atlas-agent``
or llama.cpp ``llamacpp-agent``, port 8888, text-only calls). All model
serving is Docker-hosted; no checkpoint is ever loaded in-process through
this module. Class name kept as ``VllmChatClient`` even though nothing here
calls vLLM anymore — it's just an OpenAI-compatible chat client, and
renaming would touch every caller for no behavioral change.
"""

from __future__ import annotations

import base64
import os

_DEFAULT_BASE_URL = "http://localhost:8881/v1"
_DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"


class VllmChatClient:
    """Thin wrapper around an OpenAI-compatible ``/chat/completions`` endpoint."""

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

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_params: dict[str, object] | None = None,
        extra_messages: list[dict[str, object]] | None = None,
    ) -> str:
        """Send one text-only chat-completions call and return the text response."""
        client = self._load_client()
        messages: list[dict[str, object]] = [{"role": "system", "content": system_prompt}]
        if extra_messages:
            messages.extend(extra_messages)
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        response = client.post(
            "/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                **(generation_params or {}),
            },
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
