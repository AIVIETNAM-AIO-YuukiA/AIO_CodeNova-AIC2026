"""Access to video records persisted in the run manifest."""

from __future__ import annotations

from config.settings import Experiment
from indexing.manifest import JsonlManifest
from repository.schema import VideoRecord


class VideoRepository:
    """Read video records from an experiment's ``videos.jsonl`` manifest."""

    def __init__(self, experiment: Experiment) -> None:
        self._manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")

    def list_all(self) -> list[VideoRecord]:
        """Return every recorded video."""
        return [VideoRecord.from_dict(row) for row in self._manifest.read_all()]

    def by_id(self) -> dict[str, VideoRecord]:
        """Return a ``video_id -> VideoRecord`` lookup."""
        return {video.video_id: video for video in self.list_all()}
