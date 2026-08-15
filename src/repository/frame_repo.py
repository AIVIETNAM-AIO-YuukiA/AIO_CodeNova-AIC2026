"""Access to frame records persisted in the run manifest."""

from __future__ import annotations

from config.settings import Experiment
from indexing.manifest import JsonlManifest
from repository.schema import FrameRecord


class FrameRepository:
    """Read frame records from an experiment's ``frames.jsonl`` manifest."""

    _cache_list: dict[str, list[FrameRecord]] = {}
    _cache_dict: dict[str, dict[str, FrameRecord]] = {}

    def __init__(self, experiment: Experiment) -> None:
        self._manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
        self._path_key = str(self._manifest.path)

    def list_all(self) -> list[FrameRecord]:
        """Return every recorded frame."""
        if self._path_key not in self._cache_list:
            self._cache_list[self._path_key] = [
                FrameRecord.from_dict(row) for row in self._manifest.read_all()
            ]
        return self._cache_list[self._path_key]

    def by_id(self) -> dict[str, FrameRecord]:
        """Return a ``frame_id -> FrameRecord`` lookup."""
        if self._path_key not in self._cache_dict:
            self._cache_dict[self._path_key] = {frame.frame_id: frame for frame in self.list_all()}
        return self._cache_dict[self._path_key]
