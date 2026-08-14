"""Enrich raw index hits with frame and video metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import Experiment
from core.paths import resolve_experiment_frame_path
from core.types import SearchResult
from repository import CaptionRepository, FrameRepository, VideoRepository


@dataclass(frozen=True)
class HydrationIssue:
    """Explain why an index hit could not become a usable result."""

    frame_id: str
    reason: str
    raw_path: str | None = None


@dataclass(frozen=True)
class HydrationBatch:
    """Hydrated results plus explicit diagnostics for dropped hits."""

    results: list[SearchResult]
    issues: list[HydrationIssue]


class ResultHydrator:
    """Attach frame/video metadata to index search results.

    Frame, video, and caption lookups are loaded once on construction and
    reused across queries, so serving many searches does not re-read the
    manifests each time. Captions are optional — ``by_id()`` is an empty dict
    when the experiment didn't configure the Vietnamese embedding model, so
    ``result.caption`` is simply ``None`` in that case.
    """

    def __init__(self, experiment: Experiment) -> None:
        self._experiment = experiment
        self._frames = FrameRepository(experiment).by_id()
        self._videos = VideoRepository(experiment).by_id()
        self._captions = CaptionRepository(experiment).by_id()

    def hydrate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Return results enriched with frame paths, timestamps, video names, and captions."""
        return self.hydrate_with_diagnostics(results).results

    def hydrate_with_diagnostics(self, results: list[SearchResult]) -> HydrationBatch:
        """Hydrate usable hits and report every hit that had to be dropped."""
        hydrated: list[SearchResult] = []
        issues: list[HydrationIssue] = []
        for result in results:
            frame = self._frames.get(result.frame_id)
            if frame is None:
                issues.append(HydrationIssue(result.frame_id, "FRAME_METADATA_MISSING"))
                continue
            resolution = resolve_experiment_frame_path(self._experiment, frame.frame_path)
            if not resolution.valid:
                issues.append(
                    HydrationIssue(
                        result.frame_id,
                        resolution.reason or "FRAME_PATH_INVALID",
                        resolution.raw_path,
                    )
                )
                continue
            hydrated.append(self._hydrate_one(result, str(resolution.resolved_path)))
        return HydrationBatch(hydrated, issues)

    def _hydrate_one(self, result: SearchResult, frame_path: str) -> SearchResult:
        frame = self._frames.get(result.frame_id)
        if frame is None:  # Defensive; hydrate_with_diagnostics already checked it.
            raise KeyError(result.frame_id)
        video = self._videos.get(frame.video_id)
        return SearchResult(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            score=result.score,
            frame_path=frame_path,
            video_path=video.path if video else None,
            video_name=Path(video.path).name if video else None,
            shot_id=frame.shot_id,
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            caption=self._captions.get(result.frame_id),
        )
