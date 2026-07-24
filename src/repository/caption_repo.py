"""Access to caption records persisted in the run manifest."""

from __future__ import annotations

from config.settings import Experiment
from indexing.manifest import JsonlManifest


class CaptionRepository:
    """Read VLM-generated captions from an experiment's ``captions.jsonl`` manifest.

    Captions are written by ``modules.embedding.vietnamese.VietnameseEmbedder``
    during ``embed-frames`` (see indexing/embeddings.py) and are optional: an
    experiment that doesn't configure the Vietnamese embedding model simply
    has no ``captions.jsonl``, and ``by_id()`` returns an empty mapping.
    """

    def __init__(self, experiment: Experiment) -> None:
        self._manifest = JsonlManifest(experiment.run_dir / "manifests" / "captions.jsonl")

    def by_id(self) -> dict[str, str]:
        """Return a ``frame_id -> caption`` lookup."""
        return {
            str(row["frame_id"]): str(row["caption"])
            for row in self._manifest.read_all()
            if "frame_id" in row and "caption" in row
        }
