"""Access to frame records persisted in the run manifest."""

from __future__ import annotations

from codenova.config.settings import Experiment
from codenova.indexing.manifest import JsonlManifest
from codenova.repository.schema import FrameRecord


class FrameRepository:
    """Read frame records from an experiment's ``frames.jsonl`` manifest."""

    def __init__(self, experiment: Experiment) -> None:
        self._manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")

    def list_all(self) -> list[FrameRecord]:
        """Return every recorded frame."""
        return [FrameRecord.from_dict(row) for row in self._manifest.read_all()]

    def by_id(self) -> dict[str, FrameRecord]:
        """Return a ``frame_id -> FrameRecord`` lookup."""
        return {frame.frame_id: frame for frame in self.list_all()}
