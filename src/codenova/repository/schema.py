"""Record schemas for the data access layer.

The canonical dataclasses live in :mod:`core.types`; they are re-exported here
so repositories and callers depend on ``repository.schema`` rather than reaching
across into ``core``. New persisted record types (OCR/ASR text, captions) should
be added here as the corresponding modules graduate from stubs.
"""

from __future__ import annotations

from codenova.core.types import FrameRecord, SearchResult, ShotRecord, VideoRecord

__all__ = ["FrameRecord", "SearchResult", "ShotRecord", "VideoRecord"]
