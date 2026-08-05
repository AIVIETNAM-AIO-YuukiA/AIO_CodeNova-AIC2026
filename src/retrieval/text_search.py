"""Direct full-text search over ASR / OCR — no embedding model involved.

Ported from the AIC_2025 reference project's ``/search/audio`` and
``/search/ocr`` routes: a plain BM25 query against Elasticsearch, filtered by
``source``. Unlike the KIS tracks, this never touches the GPU.
"""

from __future__ import annotations

import bisect

from config.settings import Experiment
from repository.frame_repo import FrameRepository
from retrieval.hydrator import ResultHydrator
from stores.text.factory import build_text_index
from stores.vector.base import frame_result


class NearestFrameIndex:
    """Map ``(video_id, timestamp_sec)`` to the closest known frame.

    OCR documents carry a real ``frame_id`` (one VLM call per frame), but ASR
    segments don't — speech isn't tied to a single frame — so ASR hits need
    the nearest frame in the same video looked up by timestamp instead.
    """

    def __init__(self, experiment: Experiment) -> None:
        by_video: dict[str, list[tuple[float, str]]] = {}
        for frame in FrameRepository(experiment).list_all():
            by_video.setdefault(frame.video_id, []).append((frame.timestamp_sec, frame.frame_id))
        self._by_video = {
            video_id: sorted(pairs) for video_id, pairs in by_video.items()
        }

    def nearest(self, video_id: str, timestamp_sec: float) -> str | None:
        pairs = self._by_video.get(video_id)
        if not pairs:
            return None
        times = [t for t, _ in pairs]
        pos = bisect.bisect_left(times, timestamp_sec)
        candidates = [p for p in (pos - 1, pos) if 0 <= p < len(pairs)]
        if not candidates:
            return None
        best = min(candidates, key=lambda p: abs(pairs[p][0] - timestamp_sec))
        return pairs[best][1]


def text_search(experiment: Experiment, query: str, source: str, top_k: int = 20) -> dict:
    """Search ASR (``source="asr"``) or OCR (``source="ocr"``) text directly.

    Returns matches ranked by BM25 score, hydrated with frame metadata so the
    UI can render an image + timestamp per hit like the other tracks.
    """
    query = query.strip()
    if not query:
        return {"results": [], "total": 0}

    index = build_text_index(experiment)
    docs = index.search_documents(query, top_k=top_k, source=source)

    hydrator = ResultHydrator(experiment)
    nearest_frame = NearestFrameIndex(experiment) if source == "asr" else None

    results = []
    for doc in docs:
        frame_id = doc.get("frame_id")
        timestamp_sec = doc.get("timestamp_sec")
        if not frame_id and nearest_frame is not None:
            frame_id = nearest_frame.nearest(doc.get("video_id", ""), timestamp_sec or 0.0)
        if not frame_id:
            continue

        hydrated = hydrator.hydrate([frame_result(frame_id, float(doc.get("score", 0.0)))])
        if not hydrated:
            continue
        sr = hydrated[0]
        results.append(
            {
                "frame_id": sr.frame_id,
                "video_id": sr.video_id or doc.get("video_id"),
                "video_name": sr.video_name or sr.video_id or doc.get("video_id"),
                "frame_path": sr.frame_path,
                "frame_index": sr.frame_index,
                "shot_id": sr.shot_id,
                "timestamp_sec": timestamp_sec if timestamp_sec is not None else sr.timestamp_sec,
                "score": sr.score,
                "text": doc.get("text", ""),
            }
        )

    return {"results": results, "total": len(results)}
