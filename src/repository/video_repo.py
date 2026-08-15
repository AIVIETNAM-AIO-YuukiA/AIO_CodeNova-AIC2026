"""Access to video records persisted in the run manifest."""

from __future__ import annotations

from config.settings import Experiment
from indexing.manifest import JsonlManifest
from repository.schema import VideoRecord


class VideoRepository:
    """Read video records from an experiment's ``videos.jsonl`` manifest."""

    _cache_list: dict[str, list[VideoRecord]] = {}
    _cache_dict: dict[str, dict[str, VideoRecord]] = {}

    def __init__(self, experiment: Experiment) -> None:
        self._manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
        self._path_key = str(self._manifest.path)

    def list_all(self) -> list[VideoRecord]:
        """Return every recorded video."""
        if self._path_key not in self._cache_list:
            self._cache_list[self._path_key] = [
                VideoRecord.from_dict(row) for row in self._manifest.read_all()
            ]
        return self._cache_list[self._path_key]

    def by_id(self) -> dict[str, VideoRecord]:
        """Return a ``video_id -> VideoRecord`` lookup."""
        if self._path_key not in self._cache_dict:
            self._cache_dict[self._path_key] = {video.video_id: video for video in self.list_all()}
        return self._cache_dict[self._path_key]
