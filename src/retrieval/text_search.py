"""Direct full-text search over ASR / OCR — no embedding model involved.

Ported from the AIC_2025 reference project's ``/search/audio`` and
``/search/ocr`` routes: a plain BM25 query against Elasticsearch, filtered by
``source``. Unlike the KIS tracks, this never touches the GPU.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections.abc import Iterable, Mapping
from typing import ClassVar

from config.settings import Experiment
from indexing.manifest import JsonlManifest
from repository.frame_repo import FrameRepository
from retrieval.hydrator import ResultHydrator
from stores.text.factory import build_text_index
from stores.vector.base import frame_result


DEFAULT_ASR_MAX_DURATION_SEC = 45.0
DEFAULT_ASR_PADDING_SEC = 2.0
DEFAULT_ASR_DECAY_SEC = 2.0


def infer_asr_intervals(
    documents: Iterable[Mapping[str, object]],
    max_duration_sec: float = DEFAULT_ASR_MAX_DURATION_SEC,
) -> dict[str, tuple[float, float]]:
    """Infer ``doc_id -> (start, end)`` ASR intervals from legacy records.

    Existing ASR artifacts only persist a segment's start timestamp. Within
    each video, the next segment start is therefore the best available end;
    very long gaps and the final segment are capped at ``max_duration_sec``.
    Invalid records are ignored so old partially populated manifests remain
    readable.
    """
    if max_duration_sec <= 0:
        raise ValueError("max_duration_sec must be positive")

    grouped: dict[str, list[tuple[float, int, str]]] = {}
    for position, document in enumerate(documents):
        source = document.get("source")
        if source is not None and source != "asr":
            continue
        doc_id = document.get("doc_id")
        video_id = document.get("video_id")
        if not isinstance(doc_id, str) or not doc_id or not isinstance(video_id, str):
            continue
        try:
            start_sec = float(document.get("timestamp_sec"))
        except (OverflowError, TypeError, ValueError):
            continue
        if not math.isfinite(start_sec):
            continue
        grouped.setdefault(video_id, []).append((start_sec, position, doc_id))

    intervals: dict[str, tuple[float, float]] = {}
    for segments in grouped.values():
        segments.sort(key=lambda item: (item[0], item[1]))
        for position, (start_sec, _, doc_id) in enumerate(segments):
            maximum_end = start_sec + max_duration_sec
            if position + 1 < len(segments):
                next_start = segments[position + 1][0]
                end_sec = max(start_sec, min(next_start, maximum_end))
            else:
                end_sec = maximum_end
            intervals[doc_id] = (start_sec, end_sec)
    return intervals


class NearestFrameIndex:
    """Map ``(video_id, timestamp_sec)`` to the closest known frame.

    OCR documents carry a real ``frame_id`` (one VLM call per frame), but ASR
    segments don't — speech isn't tied to a single frame — so ASR hits need
    the nearest frame in the same video looked up by timestamp instead.
    """

    _cache: ClassVar[dict[str, dict[str, list[tuple[float, str]]]]] = {}
    _metadata_cache: ClassVar[
        dict[str, dict[str, tuple[str, float | None]]]
    ] = {}

    def __init__(self, experiment: Experiment) -> None:
        run_dir_key = str(experiment.run_dir)
        if run_dir_key not in self._cache or run_dir_key not in self._metadata_cache:
            by_video: dict[str, list[tuple[float, str]]] = {}
            metadata: dict[str, tuple[str, float | None]] = {}
            for frame in FrameRepository(experiment).list_all():
                metadata[frame.frame_id] = (frame.shot_id, frame.timestamp_sec)
                if frame.timestamp_sec is None:
                    continue
                by_video.setdefault(frame.video_id, []).append(
                    (float(frame.timestamp_sec), frame.frame_id)
                )
            self._cache[run_dir_key] = {
                video_id: sorted(pairs) for video_id, pairs in by_video.items()
            }
            self._metadata_cache[run_dir_key] = metadata
        self._by_video = self._cache[run_dir_key]
        self._metadata = self._metadata_cache[run_dir_key]

    def frame_context(self, frame_id: str) -> tuple[str, float | None] | None:
        """Return cached ``(shot_id, timestamp)`` metadata for one frame."""
        return self._metadata.get(frame_id)

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

    def within(
        self, video_id: str, timestamp_sec: float, window_sec: float
    ) -> list[tuple[str, float]]:
        """Return ``(frame_id, timestamp)`` within a symmetric time window."""
        if window_sec < 0:
            raise ValueError("window_sec must be non-negative")
        pairs = self._by_video.get(video_id, [])
        if not pairs:
            return []
        times = [timestamp for timestamp, _ in pairs]
        left = bisect.bisect_left(times, timestamp_sec - window_sec)
        right = bisect.bisect_right(times, timestamp_sec + window_sec)
        return [(frame_id, timestamp) for timestamp, frame_id in pairs[left:right]]

    def nearby_weighted(
        self,
        video_id: str,
        timestamp_sec: float,
        window_sec: float = DEFAULT_ASR_PADDING_SEC,
        decay_sec: float = DEFAULT_ASR_DECAY_SEC,
    ) -> list[tuple[str, float]]:
        """Return point-in-time matches weighted by exponential distance decay."""
        if decay_sec <= 0:
            raise ValueError("decay_sec must be positive")
        return [
            (frame_id, math.exp(-abs(frame_timestamp - timestamp_sec) / decay_sec))
            for frame_id, frame_timestamp in self.within(video_id, timestamp_sec, window_sec)
        ]

    def map_interval(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float | None = None,
        padding_sec: float = DEFAULT_ASR_PADDING_SEC,
        decay_sec: float = DEFAULT_ASR_DECAY_SEC,
    ) -> list[tuple[str, float]]:
        """Map an ASR interval to nearby keyframes and temporal weights.

        Frames inside the segment receive weight 1. Frames up to
        ``padding_sec`` outside it receive ``exp(-distance / decay_sec)``.
        If no frame falls in that padded interval, the frame closest to the
        interval is returned with weight 1 to preserve legacy nearest-frame
        behavior on very sparsely sampled videos.
        """
        if padding_sec < 0:
            raise ValueError("padding_sec must be non-negative")
        if decay_sec <= 0:
            raise ValueError("decay_sec must be positive")
        if not math.isfinite(start_sec):
            raise ValueError("start_sec must be finite")
        if end_sec is None:
            end_sec = start_sec + DEFAULT_ASR_MAX_DURATION_SEC
        if not math.isfinite(end_sec):
            raise ValueError("end_sec must be finite")
        end_sec = max(end_sec, start_sec)

        pairs = self._by_video.get(video_id, [])
        if not pairs:
            return []
        times = [timestamp for timestamp, _ in pairs]
        left = bisect.bisect_left(times, start_sec - padding_sec)
        right = bisect.bisect_right(times, end_sec + padding_sec)

        weighted: list[tuple[str, float]] = []
        for timestamp, frame_id in pairs[left:right]:
            if timestamp < start_sec:
                distance = start_sec - timestamp
            elif timestamp > end_sec:
                distance = timestamp - end_sec
            else:
                distance = 0.0
            weighted.append((frame_id, math.exp(-distance / decay_sec)))
        if weighted:
            return weighted

        def distance_to_interval(pair: tuple[float, str]) -> float:
            timestamp = pair[0]
            if timestamp < start_sec:
                return start_sec - timestamp
            if timestamp > end_sec:
                return timestamp - end_sec
            return 0.0

        _, nearest_frame_id = min(pairs, key=distance_to_interval)
        return [(nearest_frame_id, 1.0)]


class AsrTemporalMapper:
    """Map legacy ASR search documents to frame IDs over inferred intervals."""

    _interval_cache: ClassVar[
        dict[str, tuple[tuple[int, int], dict[str, tuple[float, float]]]]
    ] = {}
    _interval_cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        experiment: Experiment,
        *,
        max_duration_sec: float = DEFAULT_ASR_MAX_DURATION_SEC,
        padding_sec: float = DEFAULT_ASR_PADDING_SEC,
        decay_sec: float = DEFAULT_ASR_DECAY_SEC,
    ) -> None:
        self._frames = NearestFrameIndex(experiment)
        self._max_duration_sec = max_duration_sec
        self._padding_sec = padding_sec
        self._decay_sec = decay_sec
        manifest = JsonlManifest(experiment.run_dir / "manifests" / "text.jsonl")
        cache_key = f"{manifest.path.resolve()}::{max_duration_sec}"
        try:
            stat = manifest.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            signature = (0, 0)
        with self._interval_cache_lock:
            cached = self._interval_cache.get(cache_key)
            if cached is None or cached[0] != signature:
                intervals = infer_asr_intervals(
                    manifest.read_all(), max_duration_sec=max_duration_sec
                )
                self._interval_cache[cache_key] = (signature, intervals)
            self._intervals = self._interval_cache[cache_key][1]

    def interval_for(self, document: Mapping[str, object]) -> tuple[float, float]:
        """Return an inferred interval, falling back for non-manifest documents."""
        doc_id = document.get("doc_id")
        if isinstance(doc_id, str) and doc_id in self._intervals:
            return self._intervals[doc_id]
        try:
            start_sec = float(document.get("timestamp_sec") or 0.0)
        except (OverflowError, TypeError, ValueError):
            start_sec = 0.0
        if not math.isfinite(start_sec):
            start_sec = 0.0
        return start_sec, start_sec + self._max_duration_sec

    def map_document(self, document: Mapping[str, object]) -> list[tuple[str, float]]:
        """Return temporally weighted frame candidates for one ASR document."""
        video_id = document.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            return []
        start_sec, end_sec = self.interval_for(document)
        return self._frames.map_interval(
            video_id,
            start_sec,
            end_sec,
            padding_sec=self._padding_sec,
            decay_sec=self._decay_sec,
        )


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
    asr_mapper = AsrTemporalMapper(experiment) if source == "asr" else None

    if asr_mapper is not None:
        candidates: dict[str, dict] = {}
        for doc in docs:
            start_sec, end_sec = asr_mapper.interval_for(doc)
            for frame_id, temporal_weight in asr_mapper.map_document(doc):
                score = float(doc.get("score", 0.0)) * temporal_weight
                previous = candidates.get(frame_id)
                if previous is None or score > previous["score"]:
                    candidates[frame_id] = {
                        "score": score,
                        "document": doc,
                        "temporal_weight": temporal_weight,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                    }

        ranked = sorted(candidates.items(), key=lambda item: item[1]["score"], reverse=True)[
            :top_k
        ]
        hydrated = ResultHydrator(experiment).hydrate(
            [frame_result(frame_id, candidate["score"]) for frame_id, candidate in ranked]
        )
        hydrated_by_id = {result.frame_id: result for result in hydrated}
        results = []
        for frame_id, candidate in ranked:
            sr = hydrated_by_id.get(frame_id)
            if sr is None:
                continue
            doc = candidate["document"]
            results.append(
                {
                    "frame_id": sr.frame_id,
                    "video_id": sr.video_id or doc.get("video_id"),
                    "video_name": sr.video_name or sr.video_id or doc.get("video_id"),
                    "frame_path": sr.frame_path,
                    "frame_index": sr.frame_index,
                    "shot_id": sr.shot_id,
                    # Preserve the legacy meaning: this is the transcript
                    # timestamp. The mapped frame timestamp is additive.
                    "timestamp_sec": candidate["start_sec"],
                    "frame_timestamp_sec": sr.timestamp_sec,
                    "segment_end_sec": candidate["end_sec"],
                    "temporal_weight": candidate["temporal_weight"],
                    "score": sr.score,
                    "text": doc.get("text", ""),
                }
            )
        return {"results": results, "total": len(results)}

    results = []
    for doc in docs:
        frame_id = doc.get("frame_id")
        timestamp_sec = doc.get("timestamp_sec")
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
