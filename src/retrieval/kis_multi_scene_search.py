"""KIS Multi-Scene search — delegates to TRAKE's BPJ algorithm.

Mỗi scene là một event trong TRAKE pipeline. Giữ nguyên window=15s (default).
Output giống hệt TRAKE: {"videos": [...chains...], "total_candidates": ...}
"""

from __future__ import annotations

import logging

from config.settings import Experiment

LOGGER = logging.getLogger(__name__)


def kis_multi_scene_search(
    experiment: Experiment,
    events: list[str],
    top_k: int = 300,
    window: int = 15,
) -> dict:
    """KIS Multi-Scene pipeline — wraps TRAKE's Bidirectional Pair Join.

    Mỗi scene trong query được xử lý như một event trong TRAKE.
    Output: list chains (mỗi chain = tuple N frame cho N scene, cùng video, tăng dần timestamp).

    Args:
        experiment: Experiment instance.
        events: List of scene descriptions, at least 2.
        top_k: Max chains to return (default 300).
        window: Temporal window in seconds (default 15).

    Returns:
        Dict with "videos" (chains) and "total_candidates".
    """
    from retrieval.trake_search import trake_search as _bpj_search

    return _bpj_search(
        experiment=experiment,
        events=events,
        top_k=top_k,
        window=window,
    )
