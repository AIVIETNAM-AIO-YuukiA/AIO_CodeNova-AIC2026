"""Reranker interface and factory.

Defines the base contract for second-stage rerankers and provides a factory
function to instantiate the appropriate implementation by model name.
"""

from __future__ import annotations

import logging

from core.types import SearchResult

LOGGER = logging.getLogger(__name__)


class Reranker:
    """Base interface for second-stage rerankers (e.g., BLIP-2 ITM, cross-encoder)."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Re-score and reorder candidate results for a given query."""
        raise NotImplementedError


def build_reranker(
    model_name: str | None,
    device: str = "auto",
    batch_size: int = 8,
) -> Reranker | None:
    """Instantiate a reranker by model name.

    Args:
        model_name: Short alias or HuggingFace model ID.
                    e.g. ``"blip2-itm"`` or ``"Salesforce/blip2-itm-vit-b"``.
                    Pass ``None`` to disable reranking.
        device:     Target device – ``"auto"``, ``"cuda"``, or ``"cpu"``.
        batch_size: Images per inference batch; reduce if VRAM is limited.

    Returns:
        A ``Reranker`` instance, or ``None`` if ``model_name`` is ``None``.
    """
    if model_name is None:
        return None

    from modules.reranker.blip2_itm import Blip2ItmReranker, _default_model_name

    # Short aliases resolve to BLIP2_RERANKER_MODEL (env, falling back to the
    # canonical HF ID) rather than a hardcoded ID, so the env var is the
    # single source of truth for which BLIP-2 checkpoint is used.
    _ALIASES = {"blip2-itm", "blip2-itm-vit-g"}
    resolved_name = _default_model_name() if model_name.lower() in _ALIASES else model_name
    LOGGER.info("Reranker: initializing '%s'", resolved_name)
    return Blip2ItmReranker(model_name=resolved_name, device=device, batch_size=batch_size)
