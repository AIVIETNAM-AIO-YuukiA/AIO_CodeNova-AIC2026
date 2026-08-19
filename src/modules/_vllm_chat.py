"""Shared low-level client for OpenAI-compatible chat-completions endpoints.

Used by captioning, OCR, and the agent/query-processor LLM calls. Set
``VLM_BACKEND=openrouter`` to call OpenRouter directly; otherwise requests go
to the local engine and only fall back to OpenRouter when it is unreachable.
"""

from __future__ import annotations

import base64
import logging
import os

LOGGER = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8881/v1"
_DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Transient upstream failures worth retrying (rate limit, 5xx).
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = int(os.environ.get("VLM_MAX_RETRIES", "6"))
_RETRY_BASE_SECONDS = 2.0
_RETRY_MAX_SECONDS = 60.0

# Set VLM_DISABLE_REASONING=0 to let reasoning models think (much slower).
_DISABLE_REASONING = os.environ.get("VLM_DISABLE_REASONING", "1") != "0"


def _retry_after_seconds(response) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds form) if the server sent one."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


def _sleep_before_retry(attempt: int, retry_after: float | None) -> None:
    """Honour ``Retry-After``, else exponential backoff with jitter.

    Jitter matters here: captioning runs CAPTION_WORKERS threads against the
    same endpoint, and without it every thread rate-limited in the same window
    would wake up and retry together.
    """
    import random
    import time

    if retry_after is not None:
        delay = min(retry_after, _RETRY_MAX_SECONDS)
    else:
        delay = min(_RETRY_BASE_SECONDS * (2**attempt), _RETRY_MAX_SECONDS)
    time.sleep(delay * (0.5 + random.random() * 0.5))


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
        prefer_openrouter: bool | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.model_name = model_name or os.environ.get("VLLM_MODEL", _DEFAULT_MODEL)
        self.timeout = timeout

        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if prefer_openrouter is None:
            prefer_openrouter = os.environ.get("VLM_BACKEND", "").lower() == "openrouter"
        self.prefer_openrouter = prefer_openrouter
        openrouter_base_url = openrouter_base_url or os.environ.get(
            "OPENROUTER_BASE_URL", _DEFAULT_OPENROUTER_BASE_URL
        )
        self.openrouter_base_url = openrouter_base_url.rstrip("/")
        # No default: silently falling back to some other model would change
        # what answers, so an unset OPENROUTER_MODEL must fail loudly instead.
        self.openrouter_model = openrouter_model or os.environ.get("OPENROUTER_MODEL")

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
        """POST to the configured backend. ``build_payload(model_name)`` builds the body."""
        import httpx

        if self.prefer_openrouter:
            if not self.openrouter_api_key:
                raise RuntimeError(
                    "VLM_BACKEND=openrouter but OPENROUTER_API_KEY is not set in .env."
                )
            if not self.openrouter_model:
                raise RuntimeError(
                    "VLM_BACKEND=openrouter but OPENROUTER_MODEL is not set in .env."
                )
            return self._post_openrouter(build_payload)

        client = self._load_client()
        try:
            response = client.post("/chat/completions", json=build_payload(self.model_name))
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            if not self.openrouter_api_key or not self.openrouter_model:
                raise
            LOGGER.warning(
                "Local vLLM engine at %s unreachable (%s); falling back to OpenRouter",
                self.base_url,
                exc,
            )
        return self._post_openrouter(build_payload)

    def _post_openrouter(self, build_payload):
        """POST to OpenRouter, retrying transient rate-limit/5xx responses."""
        import httpx

        client = self._load_openrouter_client()
        payload = build_payload(self.openrouter_model)
        if _DISABLE_REASONING:
            # Reasoning models spend most of their latency on thinking tokens,
            # which max_tokens does not cap. Measured on qwen3.7-flash doing
            # OCR: 16.2s / 1003 completion tokens with reasoning, 3.7s / 38
            # without. Captioning and OCR have nothing to reason about.
            payload.setdefault("reasoning", {"enabled": False})
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            retry_after = None
            try:
                response = client.post("/chat/completions", json=payload)
                if response.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES - 1:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                    retry_after = _retry_after_seconds(response)
                    LOGGER.warning(
                        "OpenRouter HTTP %s (attempt %s/%s), backing off",
                        response.status_code,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                else:
                    if response.status_code >= 400:
                        LOGGER.error("OpenRouter HTTP %s: %s", response.status_code, response.text)
                    response.raise_for_status()
                    return response
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt == _MAX_RETRIES - 1:
                    raise
            _sleep_before_retry(attempt, retry_after)
        raise last_error if last_error else RuntimeError("OpenRouter request failed")

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
