from __future__ import annotations

import logging
import os
from pathlib import Path

from core.types import SearchResult
from modules.reranker.base import Reranker

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = "Salesforce/blip2-itm-vit-g"


def _default_model_name() -> str:
    return os.environ.get("BLIP2_RERANKER_MODEL", _DEFAULT_MODEL)


class Blip2ItmReranker(Reranker):
    """Cross-encoder reranker backed by BLIP-2 Image-Text Matching.

    Unlike bi-encoders, BLIP-2 ITM processes the (image, text) pair through a
    shared Transformer, capturing fine-grained cross-modal interactions that
    independent encoders cannot model.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name or _default_model_name()
        self.device = device
        self.batch_size = batch_size
        # Lazy-loaded on first call to rerank() to avoid startup cost.
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Re-score results with BLIP-2 ITM and return them sorted by score descending.

        Args:
            query:   Text query string (e.g. ``"A red car on the highway"``).
            results: Candidate ``SearchResult`` objects; each must have a valid ``frame_path``.

        Returns:
            Results sorted by ITM match probability (highest first).
        """
        if not results:
            return results

        # Drop results whose frame file no longer exists on disk.
        # Normalize paths for cross-platform compatibility (e.g. Windows paths on Linux)
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

        model, processor, torch, device = self._load()
        itm_scores: list[float] = []

        # Process in batches to stay within GPU VRAM budget.
        for start in range(0, len(valid_results), self.batch_size):
            batch = valid_results[start : start + self.batch_size]
            itm_scores.extend(self._score_batch(batch, query, model, processor, torch, device))

        # SearchResult is a frozen dataclass; use replace() to attach the new score.
        import dataclasses

        reranked = [
            dataclasses.replace(result, score=itm_score)
            for result, itm_score in zip(valid_results, itm_scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)

        LOGGER.info(
            "Reranker: scored %d frames via BLIP-2 ITM. top-1 score=%.4f",
            len(reranked),
            reranked[0].score if reranked else 0.0,
        )
        return reranked

    def _score_batch(
        self,
        batch: list[SearchResult],
        query: str,
        model,
        processor,
        torch,
        device,
    ) -> list[float]:
        """Compute ITM match probabilities for one batch of (image, query) pairs.

        BLIP-2 ITM outputs logits of shape [B, 2]:
          dim-0 = no-match probability, dim-1 = match probability.
        Softmax converts logits to probabilities; we return dim-1 (match score).
        """
        from PIL import Image

        images = []
        try:
            for result in batch:
                normalized_path = result.frame_path.replace("\\", "/")
                images.append(Image.open(normalized_path).convert("RGB"))

            # Score each (image, text) pair individually to avoid cross-image
            # padding artifacts — BLIP-2 processor pads vision tokens to the
            # longest image in the batch, which corrupts shorter-image embeddings.
            scores: list[float] = []
            for image in images:
                inputs = processor(
                    images=image,
                    text=query,
                    return_tensors="pt",
                    truncation=True,
                ).to(device)

                if hasattr(inputs, "pixel_values") and inputs.pixel_values.dtype != model.dtype:
                    inputs["pixel_values"] = inputs.pixel_values.to(model.dtype)

                with torch.no_grad():
                    outputs = model(**inputs, use_image_text_matching_head=True)
                    # logits_per_image shape: [1, 2] — index 1 is the "match" logit.
                    prob = torch.softmax(outputs.logits_per_image, dim=-1)
                    scores.append(prob[0, 1].item())

            return scores

        finally:
            for img in images:
                img.close()

    def _load(self):
        """Lazy-load BLIP-2 processor and model onto the target device."""
        if self._model is not None:
            return self._model, self._processor, self._torch, self._device

        LOGGER.info("Reranker: loading BLIP-2 ITM model '%s'...", self.model_name)
        try:
            import torch
            from transformers import AutoProcessor, Blip2ForImageTextRetrieval
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.36 is required for BLIP-2 ITM reranking. "
                "Install with: uv add transformers"
            ) from exc

        resolved = (
            "cuda"
            if (self.device == "auto" and torch.cuda.is_available())
            else (self.device if self.device != "auto" else "cpu")
        )

        processor = AutoProcessor.from_pretrained(self.model_name)

        # Load model using accelerate's device_map to offload to RAM if VRAM is insufficient (e.g. RTX 2050 4GB)
        kwargs = {}
        if resolved == "cuda":
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        model = Blip2ForImageTextRetrieval.from_pretrained(self.model_name, **kwargs).eval()

        if resolved != "cuda":
            model = model.to(resolved)

        self._model = model
        self._processor = processor
        self._torch = torch
        self._device = resolved

        LOGGER.info("Reranker: BLIP-2 ITM loaded on device=%s", resolved)
        return model, processor, torch, resolved
