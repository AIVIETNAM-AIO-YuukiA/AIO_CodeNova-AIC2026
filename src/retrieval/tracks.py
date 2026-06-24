"""Contest retrieval track query adapters.

4 bài toán chính trong cuộc thi:
  1. Textual KIS   - Tìm video từ mô tả văn bản
  2. Video KIS     - Tìm video từ đoạn video mẫu (chưa implement)
  3. VQA           - Tìm khoảnh khắc + trả lời câu hỏi
  4. TRAKE         - Tìm danh sách sự kiện từ mô tả văn bản
"""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_TRACKS = {
    "textual_kis": "Textual KIS",
    "video_kis": "Video KIS",
    "vqa": "VQA",
    "trake": "TRAKE",
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
    if track in ("vqa", "trake"):
        parts.extend([request.context, request.query])
    elif track == "video_kis":
        parts.extend([request.query, request.context])
    else:
        parts.extend([request.query, request.context])

    retrieval_text = " ".join(part.strip() for part in parts if part and part.strip())
    if not retrieval_text:
        raise ValueError("Query text is empty.")
    return retrieval_text
