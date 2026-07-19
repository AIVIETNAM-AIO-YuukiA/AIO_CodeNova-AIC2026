"""Temporal search — tìm segment từ frame khớp query (frame-to-frame).

Dựa trên thuật toán Adaptive Temporal Search trong tài liệu training:
  - So sánh start frame với từng frame phía sau (cosine similarity)
  - Tolerance: số frame liên tiếp có similarity thấp hơn best
  - Khi tolerance == threshold → kết thúc segment

Pipeline:
  CLIP search → top-K frames → Temporal search (mỗi frame) → Segments →
  Gather shot → Shot validation → Agent (VQA) / Events (TRAKE)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class ShotInput:
    """Một shot input hoàn chỉnh mang ngữ cảnh thời gian."""

    video_id: str
    video_name: str
    frames: list[dict] = field(default_factory=list)
    frame_paths: list[str] = field(default_factory=list)
    frame_count: int = 0
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    clip_score: float = 0.0
    temporal_score: float = 0.0
    validation_score: float = 0.0
    validated: bool = False


def _norm(emb: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings."""
    return emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12)


def temporal_search_forward(
    start_idx: int,
    frame_embeddings: np.ndarray,
    tolerance_threshold: int = 3,
) -> int:
    """Dò tìm phía trước từ start_idx.

    So sánh start frame với từng frame phía sau.
    Nếu similarity không vượt quá best trong `threshold` frame liên tiếp → dừng.

    Args:
        start_idx: Vị trí frame bắt đầu.
        frame_embeddings: Ma trận [N, D] đã L2-normalize.
        tolerance_threshold: Số frame liên tiếp thấp hơn best để dừng.

    Returns:
        Chỉ mục frame cuối cùng của segment (inclusive).
    """
    n = frame_embeddings.shape[0]
    if start_idx >= n - 1:
        return start_idx

    emb = _norm(frame_embeddings)
    current_kf = emb[start_idx]
    best = -1.0
    tolerance = 0

    for idx in range(start_idx + 1, n):
        sim = float(emb[idx] @ current_kf)
        if sim >= best:
            best = sim
            tolerance = 0
        else:
            tolerance += 1
            if tolerance >= tolerance_threshold:
                return idx - tolerance_threshold
    return n - 1


def temporal_search_backward(
    start_idx: int,
    frame_embeddings: np.ndarray,
    tolerance_threshold: int = 3,
) -> int:
    """Dò tìm phía trước từ start_idx (về phía đầu).

    Args:
        start_idx: Vị trí frame bắt đầu.
        frame_embeddings: Ma trận [N, D] đã L2-normalize.
        tolerance_threshold: Số frame liên tiếp thấp hơn best để dừng.

    Returns:
        Chỉ mục frame đầu tiên của segment (inclusive).
    """
    if start_idx <= 0:
        return 0

    emb = _norm(frame_embeddings)
    current_kf = emb[start_idx]
    best = -1.0
    tolerance = 0

    for idx in range(start_idx - 1, -1, -1):
        sim = float(emb[idx] @ current_kf)
        if sim >= best:
            best = sim
            tolerance = 0
        else:
            tolerance += 1
            if tolerance >= tolerance_threshold:
                return idx + tolerance_threshold
    return 0


def find_segments(
    start_indices: list[int],
    frame_embeddings: np.ndarray,
    tolerance_threshold: int = 3,
    min_gap: int = 2,
) -> list[dict]:
    """Tìm tất cả segments từ danh sách start indices.

    1. Với mỗi start_idx: chạy forward + backward search
    2. Gộp các segment chồng lấn
    3. Tính score cho mỗi segment (dùng query similarity từ CLIP search)

    Args:
        start_indices: Danh sách vị trí frame từ CLIP search (sorted theo score).
        frame_embeddings: Ma trận [N, D] image embeddings.
        tolerance_threshold: Ngưỡng tolerance cho temporal search.
        min_gap: Khoảng cách tối thiểu giữa 2 segment để không gộp.

    Returns:
        list[dict]: Mỗi dict {start_pos, end_pos, length, center_idx}.
            Sorted theo start_pos tăng dần.
    """
    raw_segments: list[tuple[int, int, int]] = []

    for idx in start_indices:
        end_pos = temporal_search_forward(idx, frame_embeddings, tolerance_threshold)
        start_pos = temporal_search_backward(idx, frame_embeddings, tolerance_threshold)
        raw_segments.append((start_pos, end_pos, idx))

    if not raw_segments:
        return []

    # Sắp xếp theo start_pos
    raw_segments.sort(key=lambda x: x[0])

    # Gộp segment chồng lấn
    merged: list[tuple[int, int, int]] = [raw_segments[0]]
    for start, end, center in raw_segments[1:]:
        prev_start, prev_end, prev_center = merged[-1]
        if start <= prev_end + min_gap:
            new_start = min(prev_start, start)
            new_end = max(prev_end, end)
            merged[-1] = (new_start, new_end, prev_center)
        else:
            merged.append((start, end, center))

    return [
        {
            "start_pos": int(s),
            "end_pos": int(e),
            "length": int(e - s + 1),
            "center_idx": int(c),
        }
        for s, e, c in merged
    ]


def gather_frame_s(
    segment: dict,
    frame_records: list[dict],
) -> ShotInput | None:
    """Gom frame từ segment thành ShotInput.

    Args:
        segment: Dict {start_pos, end_pos, center_idx, length} từ find_segments.
        frame_records: Danh sách metadata frame (sorted theo thời gian).

    Returns:
        ShotInput hoặc None nếu không hợp lệ.
    """
    if segment is None:
        return None

    start = segment["start_pos"]
    end = segment["end_pos"]
    end = min(end + 1, len(frame_records))

    selected = frame_records[start:end]
    if not selected:
        return None

    video_id = selected[0].get("video_id", "unknown")
    frame_paths = [str(r.get("frame_path", "")) for r in selected]

    timestamps = [r.get("timestamp_sec") for r in selected if r.get("timestamp_sec") is not None]

    LOGGER.info(
        "gather_frame_s: video=%s start=%d end=%d frames=%d",
        video_id,
        segment["start_pos"],
        segment["end_pos"],
        len(selected),
    )

    return ShotInput(
        video_id=video_id,
        video_name="",
        frames=selected,
        frame_paths=frame_paths,
        frame_count=len(selected),
        start_timestamp=timestamps[0] if timestamps else None,
        end_timestamp=timestamps[-1] if timestamps else None,
        clip_score=0.0,
    )


def load_temporal_data(
    run_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    """Load embeddings và metadata frame cho temporal search.

    Args:
        run_dir: Thư mục experiment (runs/<experiment-name>/).

    Returns:
        (frame_embeddings, frame_metadata_list)
        - frame_embeddings: np.ndarray shape [N, D] sorted theo frame_index.
        - frame_metadata_list: list[dict] mỗi phần tử chứa frame_id, video_id,
          shot_id, frame_index, timestamp_sec, frame_path.
    """
    embeddings_path = run_dir / "embeddings" / "frames.npz"
    frame_ids_path = run_dir / "embeddings" / "frame_ids.json"
    frames_manifest = run_dir / "manifests" / "frames.jsonl"

    if not embeddings_path.exists() or not frame_ids_path.exists():
        raise FileNotFoundError(
            f"Chưa có embeddings. Chạy 'embed-frames' trước. " f"Missing: {embeddings_path}"
        )

    vectors = np.load(embeddings_path)["embeddings"].astype("float32")
    id_list = json.loads(frame_ids_path.read_text(encoding="utf-8"))

    frame_map: dict[str, dict] = {}
    if frames_manifest.exists():
        with open(frames_manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    frame_map[row["frame_id"]] = row

    records = []
    for fid in id_list:
        meta = frame_map.get(fid, {})
        if meta.get("frame_id") is None:
            meta["frame_id"] = fid
        records.append(meta)

    sorted_indices = sorted(
        range(len(records)),
        key=lambda i: (
            records[i].get("video_id", ""),
            records[i].get("frame_index", 0) or 0,
        ),
    )

    sorted_vectors = vectors[sorted_indices]
    sorted_records = [records[i] for i in sorted_indices]

    return sorted_vectors, sorted_records


class ShotValidator:
    """Xác thực shot trước khi chuyển cho VLM Agent.

    Dựa trên slide:
      Group frames by shot → Average scores → Sort → TopK Result
    """

    def __init__(
        self,
        min_frames: int = 2,
        min_clip_score: float = 0.15,
        temporal_weight: float = 0.3,
    ) -> None:
        self.min_frames = min_frames
        self.min_clip_score = min_clip_score
        self.temporal_weight = temporal_weight

    def validate(
        self,
        shot: ShotInput,
        query_embedding: np.ndarray,
        all_embeddings: np.ndarray,
        all_records: list[dict],
    ) -> ShotInput:
        """Đánh giá shot và cập nhật validation_score.

        Args:
            shot: ShotInput cần validate.
            query_embedding: CLIP text embedding của query.
            all_embeddings: Tất cả frame embeddings [N x D].
            all_records: Tất cả frame records (cùng thứ tự với embeddings).

        Returns:
            ShotInput với validation_score và validated flag.
        """
        if shot.frame_count < self.min_frames:
            shot.validation_score = 0.0
            shot.validated = False
            LOGGER.warning("Shot too few frames (%d < %d)", shot.frame_count, self.min_frames)
            return shot

        q = np.asarray(query_embedding, dtype="float32").flatten()
        q = q / (np.linalg.norm(q) + 1e-12)

        frame_ids_in_shot = {f.get("frame_id") for f in shot.frames if f.get("frame_id")}
        indices = [
            i for i, rec in enumerate(all_records) if rec.get("frame_id") in frame_ids_in_shot
        ]

        if not indices:
            LOGGER.warning("No matching embeddings found for shot frames")
            shot.validation_score = 0.0
            shot.validated = False
            return shot

        # 1. CLIP score trung bình của shot với query
        shot_embs = all_embeddings[indices]
        shot_embs = shot_embs / (np.linalg.norm(shot_embs, axis=1, keepdims=True) + 1e-12)
        clip_sims = shot_embs @ q
        avg_clip = float(np.mean(clip_sims))

        # 2. Temporal consistency
        timestamps = [
            all_records[i].get("timestamp_sec") or all_records[i].get("frame_index", 0)
            for i in indices
        ]
        temporal_gaps = [abs(timestamps[j + 1] - timestamps[j]) for j in range(len(timestamps) - 1)]
        temporal_consistency = 1.0
        if temporal_gaps:
            max_gap = max(temporal_gaps)
            temporal_consistency = 1.0 / (1.0 + max_gap) if max_gap > 0 else 1.0

        # 3. Composite score
        composite = (
            1 - self.temporal_weight
        ) * avg_clip + self.temporal_weight * temporal_consistency

        shot.clip_score = avg_clip
        shot.temporal_score = temporal_consistency
        shot.validation_score = composite
        shot.validated = composite >= self.min_clip_score

        LOGGER.info(
            "Shot validation: frames=%d clip=%.4f temporal=%.4f composite=%.4f validated=%s",
            shot.frame_count,
            avg_clip,
            temporal_consistency,
            composite,
            shot.validated,
        )

        return shot
