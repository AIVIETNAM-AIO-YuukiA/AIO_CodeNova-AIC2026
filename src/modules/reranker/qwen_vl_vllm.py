"""Cross-encoder reranker backed by Qwen3-VL-Reranker-2B, served over HTTP by vLLM.

Unlike BLIP-2 ITM (loaded in-process via transformers), this model runs as a
separate vLLM container — see docker-compose.yml's ``vllm-reranker`` service
and Makefile's ``vllm-reranker-up`` — so the 2B-parameter model's VRAM lives
on whatever GPU hosts that container, not the process running the UI/agent.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from core.types import SearchResult
from modules.reranker.base import Reranker

LOGGER = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8884"
_DEFAULT_MODEL = "Qwen/Qwen3-VL-Reranker-2B"


def _default_base_url() -> str:
    return os.environ.get("QWEN_VL_RERANKER_URL", _DEFAULT_BASE_URL)


def _default_model_name() -> str:
    return os.environ.get("QWEN_VL_RERANKER_MODEL", _DEFAULT_MODEL)


class QwenVlVllmReranker(Reranker):
    """Cross-encoder reranker calling a vLLM ``/v1/score`` endpoint per batch.

    Scoring strategy (hybrid): same blend as ``Blip2ItmReranker`` — the raw
    cross-encoder score is combined with the first-stage CLIP-family score so
    a reranker having an off day cannot fully override an already well
    calibrated embedding score.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 30.0,
        batch_size: int = 16,
        score_weight: float = 0.6,
    ) -> None:
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.model_name = model_name or _default_model_name()
        self.timeout = timeout
        self.batch_size = batch_size
        self.score_weight = score_weight
        self._client = None

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Re-score results with the vLLM-hosted reranker, sorted descending."""
        if not results:
            return results

        valid_results = []
        for r in results:
            if r.frame_path:
                normalized_path = r.frame_path.replace("\\", "/")
                if Path(normalized_path).exists():
                    valid_results.append(r)

        missing = len(results) - len(valid_results)
        if missing > 0:
            LOGGER.warning("Reranker: skipping %d results with missing frame_path.", missing)
        if not valid_results:
            LOGGER.warning("Reranker: no valid frames found; returning original ranking.")
            return results

        scores: list[float] = []
        for start in range(0, len(valid_results), self.batch_size):
            batch = valid_results[start : start + self.batch_size]
            scores.extend(self._score_batch(batch, query))

        import dataclasses

        clip_scores = [r.score for r in valid_results]
        max_clip = max(clip_scores) if clip_scores else 1.0
        min_clip = min(clip_scores) if clip_scores else 0.0
        clip_range = max_clip - min_clip or 1.0

        reranked = []
        for result, score in zip(valid_results, scores):
            clip_norm = (result.score - min_clip) / clip_range
            hybrid = self.score_weight * score + (1.0 - self.score_weight) * clip_norm
            reranked.append(dataclasses.replace(result, score=hybrid))

        reranked.sort(key=lambda r: r.score, reverse=True)
        LOGGER.info(
            "Reranker: scored %d frames via Qwen3-VL vLLM (score_weight=%.2f). top-1 hybrid=%.4f",
            len(reranked),
            self.score_weight,
            reranked[0].score if reranked else 0.0,
        )
        return reranked

    def _score_batch(self, batch: list[SearchResult], query: str) -> list[float]:
        """POST one batch of (image, query) pairs to vLLM's /v1/score endpoint.

        Multimodal documents must be wrapped in vLLM's ``ScoreMultiModalParam``
        shape (``{"content": [...]}``) — a bare content-part object (the old
        ``{"type": "image_url", ...}`` without the wrapper) fails schema
        validation with a 400.
        """
        client = self._load_client()

        documents = []
        for result in batch:
            normalized_path = result.frame_path.replace("\\", "/")
            image_data_url = _encode_image_data_url(normalized_path)
            documents.append(
                {"content": [{"type": "image_url", "image_url": {"url": image_data_url}}]}
            )

        response = client.post(
            "/v1/score",
            json={
                "model": self.model_name,
                "queries": query,
                "documents": documents,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return [float(item["score"]) for item in payload["data"]]

    def _load_client(self):
        if self._client is not None:
            return self._client
        import httpx

        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client


def _encode_image_data_url(image_path: str) -> str:
    suffix = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpg"
    mime = "image/png" if suffix == "png" else "image/jpeg"
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
