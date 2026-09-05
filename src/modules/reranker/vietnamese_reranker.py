"""Text-text reranker backed by AITeamVN/Vietnamese_Reranker (BGE-reranker-v2-m3 based).

Complements ``Blip2ItmReranker``: BLIP-2 ITM scores (query text, keyframe
image) pairs via cross-attention; this reranker scores (query text, keyframe
caption text) pairs via a cross-encoder sequence-classification head. The two
signals cover different failure modes — BLIP-2 catches visual mismatches a
caption might gloss over, this reranker catches text details (on-screen
ticker/banner content, named entities) that a caption captures but the
visual embedder cannot. See ``retrieval/fusion.py`` for how the two scores
are combined.

Requires each ``SearchResult`` to have already been hydrated with a
``caption`` (see ``retrieval/hydrator.py``); results without one are skipped,
matching ``Blip2ItmReranker``'s handling of missing ``frame_path``.
"""

from __future__ import annotations

import dataclasses
import logging
import os

from core.types import SearchResult
from modules.reranker.base import Reranker

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = "AITeamVN/Vietnamese_Reranker"


class VietnameseReranker(Reranker):
    """Cross-encoder reranker backed by a Vietnamese (query, caption) relevance model."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name or os.environ.get("VIETNAMESE_RERANKER_MODEL", _DEFAULT_MODEL)
        self.device = device
        self.batch_size = batch_size
        # Lazy-loaded on first call to rerank() to avoid startup cost.
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._resolved_device = None

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Re-score results with the Vietnamese cross-encoder and return them sorted.

        Args:
            query:   Text query string (Vietnamese).
            results: Candidate ``SearchResult`` objects; each must carry a
                     ``caption`` attribute (attached during hydration).

        Returns:
            Results sorted by relevance score descending.
        """
        if not results:
            return results

        valid_results = [r for r in results if r.caption]
        missing = len(results) - len(valid_results)
        if missing > 0:
            LOGGER.warning("Reranker: skipping %d results with no caption.", missing)
        if not valid_results:
            LOGGER.warning("Reranker: no captioned results; returning original ranking.")
            return results

        model, tokenizer, torch, device = self._load()
        scores: list[float] = []
        for start in range(0, len(valid_results), self.batch_size):
            batch = valid_results[start : start + self.batch_size]
            scores.extend(self._score_batch(batch, query, model, tokenizer, torch, device))

        reranked = [
            dataclasses.replace(result, score=score) for result, score in zip(valid_results, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)

        LOGGER.info(
            "Reranker: scored %d frames via Vietnamese_Reranker. top-1 score=%.4f",
            len(reranked),
            reranked[0].score if reranked else 0.0,
        )
        return reranked

    def _score_batch(
        self,
        batch: list[SearchResult],
        query: str,
        model,
        tokenizer,
        torch,
        device,
    ) -> list[float]:
        pairs = [[query, result.caption] for result in batch]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=2304,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.view(-1).float()
            scores = torch.sigmoid(logits)
        return scores.detach().cpu().tolist()

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer, self._torch, self._resolved_device

        LOGGER.info("Reranker: loading Vietnamese_Reranker model '%s'...", self.model_name)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Vietnamese_Reranker. "
                "Install with: uv add transformers"
            ) from exc

        resolved = (
            "cuda"
            if (self.device == "auto" and torch.cuda.is_available())
            else (self.device if self.device != "auto" else "cpu")
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name).eval().to(resolved)
        )

        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._resolved_device = resolved

        LOGGER.info("Reranker: Vietnamese_Reranker loaded on device=%s", resolved)
        return model, tokenizer, torch, resolved
