"""Shared low-level client for the OpenRouter chat-completions endpoint.

Used by captioning, OCR, and the agent/query-processor LLM calls.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger(__name__)

_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Transient upstream failures worth retrying (rate limit, 5xx).
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = int(os.environ.get("VLM_MAX_RETRIES", "6"))
_RETRY_BASE_SECONDS = 2.0
_RETRY_MAX_SECONDS = 60.0

# Set VLM_DISABLE_REASONING=0 to let reasoning models think (much slower).
_DISABLE_REASONING = os.environ.get("VLM_DISABLE_REASONING", "1") != "0"

_MAX_IMAGES_PER_REQUEST = 6
_IMAGE_DETAIL_VALUES = frozenset({"auto", "low", "high"})


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
    """Thin wrapper around OpenRouter's OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        timeout: float = 60.0,
        openrouter_base_url: str | None = None,
        openrouter_model: str | None = None,
        openrouter_api_key: str | None = None,
        openrouter_provider: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = None if max_retries is None else max(0, int(max_retries))
        self._usage_local = threading.local()
        self.last_usage: dict[str, object] = {}

        raw_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        # ``source .env`` on a CRLF file can leave a trailing carriage return
        # in the exported value. It is invisible in shell output but makes an
        # otherwise valid bearer token fail with HTTP 401.
        self.openrouter_api_key = raw_api_key.strip() if raw_api_key else raw_api_key
        openrouter_base_url = openrouter_base_url or os.environ.get(
            "OPENROUTER_BASE_URL", _DEFAULT_OPENROUTER_BASE_URL
        )
        self.openrouter_base_url = openrouter_base_url.strip().rstrip("/")
        # No default: silently falling back to some other model would change
        # what answers, so an unset OPENROUTER_MODEL must fail loudly instead.
        raw_model = openrouter_model or os.environ.get("OPENROUTER_MODEL")
        self.openrouter_model = raw_model.strip() if raw_model else raw_model
        # Pin the upstream provider (e.g. "relace") instead of letting
        # OpenRouter auto-route across whichever providers host the model.
        # No env fallback here on purpose: callers pass this explicitly per
        # use case (e.g. query_processor.py's OPENROUTER_PROVIDER_FOR_CHAT)
        # rather than one setting silently pinning every model on this client
        # (OCR/captioning included) to a provider that may not host them.
        self.openrouter_provider = (
            openrouter_provider.strip() if openrouter_provider else openrouter_provider
        )

        self._openrouter_client = None
        self._openrouter_client_lock = threading.Lock()

    @property
    def last_usage(self) -> dict[str, object]:
        """Return usage for the current request thread only."""
        return getattr(self._usage_local, "value", {})

    @last_usage.setter
    def last_usage(self, value: dict[str, object]) -> None:
        self._usage_local.value = dict(value) if isinstance(value, dict) else {}

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

        response = self._post_openrouter(
            lambda model_name: {"model": model_name, "messages": messages, **generation_params}
        )
        payload = response.json()
        self._store_response_usage(payload)
        return payload["choices"][0]["message"]["content"].strip()

    def complete_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | os.PathLike[str]],
        generation_params: dict[str, object] | None = None,
        extra_messages: list[dict[str, object]] | None = None,
        image_labels: Sequence[str] | None = None,
        detail: str | None = "high",
    ) -> str:
        """Send an ordered, labelled set of images in one completion request.

        Images are represented as alternating label and ``image_url`` content
        blocks.  This keeps labels such as ``F1 @ 12.3s`` adjacent to the image
        they identify, which is important when the caller later asks the model
        to cite supporting frames.
        """
        paths = [Path(image_path) for image_path in image_paths]
        if not paths:
            raise ValueError("complete_with_images requires at least one image")
        if len(paths) > _MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"complete_with_images accepts at most {_MAX_IMAGES_PER_REQUEST} images"
            )

        if image_labels is None:
            labels = [f"F{index}" for index in range(1, len(paths) + 1)]
        else:
            labels = [str(label).strip() for label in image_labels]
            if len(labels) != len(paths):
                raise ValueError("image_labels must contain exactly one label per image")
            if any(not label for label in labels):
                raise ValueError("image labels must not be empty")

        if detail is not None and detail not in _IMAGE_DETAIL_VALUES:
            allowed = ", ".join(sorted(_IMAGE_DETAIL_VALUES))
            raise ValueError(f"detail must be one of {allowed}, or None")

        missing_paths = [str(path) for path in paths if not path.is_file()]
        if missing_paths:
            missing = ", ".join(missing_paths)
            raise FileNotFoundError(f"Image file(s) not found: {missing}")

        content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]
        for label, path in zip(labels, paths, strict=True):
            content.append({"type": "text", "text": f"[{label}]"})
            image_url: dict[str, object] = {"url": _encode_image_data_url(str(path))}
            if detail is not None:
                image_url["detail"] = detail
            content.append({"type": "image_url", "image_url": image_url})

        messages: list[dict[str, object]] = [{"role": "system", "content": system_prompt}]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": content})

        response = self._post_openrouter(
            lambda model_name: {
                "model": model_name,
                "messages": messages,
                **(generation_params or {}),
            }
        )
        payload = response.json()
        self._store_response_usage(payload)
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

        response = self._post_openrouter(
            lambda model_name: {
                "model": model_name,
                "messages": messages,
                **(generation_params or {}),
            }
        )
        payload = response.json()
        self._store_response_usage(payload)
        return payload["choices"][0]["message"]["content"].strip()

    def _store_response_usage(self, payload: dict[str, object]) -> None:
        """Keep provider token usage together with the real HTTP attempt count."""
        previous = self.last_usage
        request_count = previous.get("request_count") if isinstance(previous, dict) else None
        provider_usage = payload.get("usage")
        usage = dict(provider_usage) if isinstance(provider_usage, dict) else {}
        if isinstance(request_count, int) and request_count > 0:
            usage["request_count"] = request_count
        self.last_usage = usage

    def _post_openrouter(self, build_payload):
        """POST to OpenRouter, retrying transient rate-limit/5xx responses."""
        import httpx

        # Thread-local because one shared client can verify VQA candidates in
        # parallel. A missing key/model or image validation failure therefore
        # correctly reports zero HTTP requests rather than one logical call.
        self.last_usage = {"request_count": 0}
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in .env.")
        if not self.openrouter_model:
            raise RuntimeError("OPENROUTER_MODEL is not set in .env.")

        client = self._load_openrouter_client()
        payload = build_payload(self.openrouter_model)
        if self.openrouter_provider:
            payload.setdefault(
                "provider", {"order": [self.openrouter_provider], "allow_fallbacks": False}
            )
        if _DISABLE_REASONING:
            # Reasoning models spend most of their latency on thinking tokens,
            # which max_tokens does not cap. Measured on qwen3.7-flash doing
            # OCR: 16.2s / 1003 completion tokens with reasoning, 3.7s / 38
            # without. Captioning and OCR have nothing to reason about.
            payload.setdefault("reasoning", {"enabled": False})
        last_error: Exception | None = None
        max_attempts = _MAX_RETRIES if self.max_retries is None else self.max_retries + 1
        for attempt in range(max_attempts):
            retry_after = None
            try:
                self.last_usage = {"request_count": attempt + 1}
                response = client.post("/chat/completions", json=payload)
                if response.status_code in _RETRY_STATUS and attempt < max_attempts - 1:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                    retry_after = _retry_after_seconds(response)
                    LOGGER.warning(
                        "OpenRouter HTTP %s (attempt %s/%s), backing off",
                        response.status_code,
                        attempt + 1,
                        max_attempts,
                    )
                else:
                    if response.status_code >= 400:
                        LOGGER.error("OpenRouter HTTP %s: %s", response.status_code, response.text)
                    if response.status_code == 401:
                        raise RuntimeError(
                            "OpenRouter rejected OPENROUTER_API_KEY (HTTP 401). "
                            "Verify the key and restart serve-ui after changing .env."
                        )
                    response.raise_for_status()
                    return response
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    raise
            _sleep_before_retry(attempt, retry_after)
        raise last_error if last_error else RuntimeError("OpenRouter request failed")

    def _load_openrouter_client(self):
        if self._openrouter_client is not None:
            return self._openrouter_client
        with self._openrouter_client_lock:
            if self._openrouter_client is not None:
                return self._openrouter_client
            import httpx

            limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
            headers = {"Authorization": f"Bearer {self.openrouter_api_key}"}
            self._openrouter_client = httpx.Client(
                base_url=self.openrouter_base_url,
                timeout=self.timeout,
                limits=limits,
                headers=headers,
            )
        return self._openrouter_client


def _encode_image_data_url(image_path: str) -> str:
    """Read an image file and return it as a base64 data URL."""
    suffix = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpg"
    mime = "image/png" if suffix == "png" else "image/jpeg"
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
