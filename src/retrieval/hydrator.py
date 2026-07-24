"""Enrich raw index hits with frame and video metadata."""

from __future__ import annotations

from pathlib import Path

from config.settings import Experiment
from core.types import SearchResult
from repository import CaptionRepository, FrameRepository, VideoRepository


class ResultHydrator:
    """Attach frame/video metadata to index search results.

    Frame, video, and caption lookups are loaded once on construction and
    reused across queries, so serving many searches does not re-read the
    manifests each time. Captions are optional — ``by_id()`` is an empty dict
    when the experiment didn't configure the Vietnamese embedding model, so
    ``result.caption`` is simply ``None`` in that case.
    """

    def __init__(self, experiment: Experiment) -> None:
        self._frames = FrameRepository(experiment).by_id()
        self._videos = VideoRepository(experiment).by_id()
        self._captions = CaptionRepository(experiment).by_id()

    def hydrate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Return results enriched with frame paths, timestamps, video names, and captions."""
        return [self._hydrate_one(result) for result in results]

    def _hydrate_one(self, result: SearchResult) -> SearchResult:
        frame = self._frames.get(result.frame_id)
        if frame is None:
            return result
        video = self._videos.get(frame.video_id)
        return SearchResult(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            score=result.score,
            frame_path=frame.frame_path,
            video_path=video.path if video else None,
            video_name=Path(video.path).name if video else None,
            shot_id=frame.shot_id,
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            caption=self._captions.get(result.frame_id),
        )
