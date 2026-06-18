"""Contest retrieval track query adapters."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_TRACKS = {
    "textual_kis": "Textual KIS",
    "vqa": "VQA",
    "qa": "Question Answering",
    "visual_kis": "Visual KIS",
}


@dataclass(frozen=True)
class TrackQuery:
    """Structured retrieval request from the UI."""

    track: str
    query: str = ""
    question: str = ""
    context: str = ""


def build_retrieval_text(request: TrackQuery) -> str:
    """Build the current CLIP text query for a contest track.

    This is intentionally conservative: until the backend has separate VQA and
    multi-modal modules, all tracks are routed to the existing text-to-frame
    retrieval index with track-specific text composition.
    """
    track = request.track.strip().lower()
    if track not in SUPPORTED_TRACKS:
        raise ValueError(f"Unsupported retrieval track: {request.track}")

    parts = []
    if track == "vqa":
        parts.extend([request.context, request.question, request.query])
    elif track == "qa":
        parts.extend([request.question, request.context, request.query])
    else:
        parts.extend([request.query, request.context, request.question])

    retrieval_text = " ".join(part.strip() for part in parts if part and part.strip())
    if not retrieval_text:
        raise ValueError("Query text is empty.")
    return retrieval_text
