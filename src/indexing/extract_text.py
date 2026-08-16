"""OCR + ASR text extraction stage — offline, indexes into Elasticsearch.

Two independent sub-stages sharing one job-state table (``EXTRACT_OCR`` /
``EXTRACT_ASR``), each resumable on its own. Every document is also appended
to ``manifests/text.jsonl`` as it is produced, so losing the Elasticsearch
volume never means re-running OCR/ASR — ``import-text`` reloads from there.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
import threading

from core.errors import CodeNovaError
from core.logging import get_logger
from core.paths import require_experiment_frame_path
from core.types import FrameRecord, VideoRecord
from indexing.manifest import JsonlManifest
from indexing.partitions import PartitionStore
from indexing.state import JobState
from modules.asr.gipformer import GipformerAsrModel
from modules.ocr.vllm import VllmOcrModel
from stores.text.base import TextDocument
from stores.text.factory import build_text_index

LOGGER = get_logger(__name__)

_DEFAULT_OCR_WORKERS = 8

# Set OCR_ONE_FRAME_PER_SHOT=1 to OCR a single representative frame per shot
# instead of every keyframe. On-screen text is usually static within a shot, so
# this cuts VLM calls (and cost) by ~2/3 at the risk of missing text that
# appears only mid-shot, such as a moving ticker.
_ONE_FRAME_PER_SHOT = os.environ.get("OCR_ONE_FRAME_PER_SHOT", "0").lower() not in (
    "0",
    "false",
    "",
)


def _pick_one_frame_per_shot(frames: list[FrameRecord]) -> list[FrameRecord]:
    """Keep the middle frame of each shot, which is furthest from cut boundaries."""
    by_shot: dict[str, list[FrameRecord]] = {}
    for frame in frames:
        by_shot.setdefault(frame.shot_id, []).append(frame)
    picked = []
    for shot_frames in by_shot.values():
        shot_frames.sort(key=lambda f: f.timestamp_sec)
        picked.append(shot_frames[len(shot_frames) // 2])
    picked.sort(key=lambda f: (f.video_id, f.timestamp_sec))
    return picked


class _TextSink:
    """Writes documents to the text index and mirrors them into text.jsonl.

    A video whose ASR/OCR finished writing documents and then crashed before
    ``JobState`` recorded it as ``COMPLETED`` gets reprocessed on the next
    run. Rather than dedupe against an in-memory set (which only protects a
    single run), each video's documents are kept in their own partition file
    under ``manifests/partitions/text/<source>/<video_id>.jsonl``, rewritten
    in full (keyed by ``doc_id``) every time that video produces documents —
    so a rerun, even from a different process, converges on the same content
    instead of appending duplicates. ``text.jsonl`` is then rebuilt from all
    partitions on every write, which is the source of truth on disk;
    Elasticsearch is a derived index, upserted after the JSONL is durable.
    """

    def __init__(self, experiment) -> None:
        self._index = build_text_index(experiment)
        self._manifest = JsonlManifest(experiment.run_dir / "manifests" / "text.jsonl")
        self._partition_root = experiment.run_dir / "manifests" / "partitions" / "text"
        self._lock = threading.Lock()

    def write(self, documents: list[TextDocument]) -> None:
        if not documents:
            return
        with self._lock:
            grouped: dict[tuple[str, str], list[TextDocument]] = {}
            for document in documents:
                grouped.setdefault((document.source, document.video_id), []).append(document)
            for (source, video_id), group in grouped.items():
                store = PartitionStore(self._partition_root / source)
                existing = store.path_for(video_id)
                by_id = (
                    {
                        str(row["doc_id"]): row
                        for row in JsonlManifest(existing).read_all(strict=True)
                    }
                    if existing.exists()
                    else {}
                )
                by_id.update({document.doc_id: asdict(document) for document in group})
                store.write(video_id, by_id.values(), partition_key="video_id", unique_key="doc_id")

            rows: list[dict[str, object]] = []
            for path in sorted(self._partition_root.glob("*/*.jsonl")):
                rows.extend(JsonlManifest(path).read_all(strict=True))
            self._manifest.replace_all(rows, unique_key="doc_id", backup=True)
            # Elasticsearch is a derived index. Publish durable JSONL first,
            # then upsert the same documents into the online text store.
            self._index.index_documents(documents)


def extract_ocr(experiment, force: bool = False) -> int:
    """Run OCR on every keyframe, indexing results and mirroring them to JSONL.

    Frames with no on-screen text are still marked completed so they aren't
    retried, but produce no document. Runs CAPTION_WORKERS threads since each
    call is a slow VLM round trip.
    """
    frames_manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    frames = [FrameRecord.from_dict(row) for row in frames_manifest.read_all()]
    frames = [f for f in frames if not state.should_skip(f.frame_id, "EXTRACT_OCR", force=force)]
    if _ONE_FRAME_PER_SHOT:
        before = len(frames)
        frames = _pick_one_frame_per_shot(frames)
        LOGGER.info("OCR one-frame-per-shot: %s frames -> %s shots", before, len(frames))
    if not frames:
        LOGGER.warning("No frames to OCR (all done, or none found)")
        return 0

    ocr_model = VllmOcrModel()
    sink = _TextSink(experiment)
    workers = int(os.environ.get("CAPTION_WORKERS", _DEFAULT_OCR_WORKERS))

    def _process(frame: FrameRecord) -> tuple[FrameRecord, TextDocument | None, Exception | None]:
        try:
            text = ocr_model.recognize(
                str(require_experiment_frame_path(experiment, frame.frame_path))
            )
            document = (
                TextDocument(
                    doc_id=f"{frame.frame_id}__ocr",
                    video_id=frame.video_id,
                    frame_id=frame.frame_id,
                    text=text,
                    source="ocr",
                    timestamp_sec=frame.timestamp_sec,
                )
                if text
                else None
            )
            return frame, document, None
        except Exception as exc:
            LOGGER.exception("OCR failed frame_id=%s", frame.frame_id)
            state.mark(frame.frame_id, "EXTRACT_OCR", "FAILED", error=str(exc))
            return frame, None, exc

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process, frames))

    documents = [document for _, document, error in results if error is None and document]
    if documents:
        sink.write(documents)
    for frame, document, error in results:
        if error is None:
            state.mark(
                frame.frame_id,
                "EXTRACT_OCR",
                "COMPLETED" if document else "COMPLETED_NO_OUTPUT",
            )
    indexed = len(documents)
    LOGGER.info("OCR extraction complete: %s frames indexed", indexed)
    return indexed


def extract_asr(experiment, force: bool = False) -> int:
    """Run ASR on every video's audio track and index the segments into Elasticsearch.

    Returns the number of transcript segments newly indexed. Videos with no
    audio stream produce zero segments but are still marked completed (see
    ``GipformerAsrModel.transcribe`` — that's a valid input, not a failure).
    """
    videos_manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    videos = [VideoRecord.from_dict(row) for row in videos_manifest.read_all()]
    if not videos:
        LOGGER.warning("No videos found for ASR extraction")
        return 0

    asr_model = GipformerAsrModel()
    sink = _TextSink(experiment)
    indexed = 0

    for position, video in enumerate(videos, start=1):
        if state.should_skip(video.video_id, "EXTRACT_ASR", force=force):
            continue
        try:
            LOGGER.info(
                "[video %s/%s] ASR started video_id=%s path=%s",
                position,
                len(videos),
                video.video_id,
                video.path,
            )
            transcripts = asr_model.transcribe(video.path)
            documents = [
                TextDocument(
                    doc_id=f"{video.video_id}__asr__{i:04d}",
                    video_id=video.video_id,
                    frame_id=None,
                    text=t.text,
                    source="asr",
                    timestamp_sec=t.start_time_sec,
                )
                for i, t in enumerate(transcripts)
                if t.text
            ]
            if documents:
                sink.write(documents)
                indexed += len(documents)
            state.mark(
                video.video_id,
                "EXTRACT_ASR",
                "COMPLETED" if documents else "COMPLETED_NO_OUTPUT",
            )
            LOGGER.info(
                "[video %s/%s] ASR completed video_id=%s segments=%s",
                position,
                len(videos),
                video.video_id,
                len(documents),
            )
        except Exception as exc:
            LOGGER.exception("ASR failed video_id=%s", video.video_id)
            state.mark(video.video_id, "EXTRACT_ASR", "FAILED", error=str(exc))

    LOGGER.info("ASR extraction complete: %s segments indexed", indexed)
    return indexed


def export_text(experiment) -> int:
    """Re-dump every OCR/ASR document from Elasticsearch to ``manifests/text.jsonl``.

    Extraction already mirrors documents there as they are produced; this
    rebuilds the file from Elasticsearch when it has drifted or been deleted.
    """
    text_index = build_text_index(experiment)
    output_path = experiment.run_dir / "manifests" / "text.jsonl"
    manifest = JsonlManifest(output_path)
    output_path.write_text("", encoding="utf-8")  # start fresh; this is a full re-export

    count = 0
    for doc in text_index.export_all():
        manifest.append(doc)
        count += 1

    LOGGER.info("Exported %s text documents to %s", count, output_path)
    return count


def import_text(experiment) -> int:
    """Load ``manifests/text.jsonl`` back into Elasticsearch.

    The counterpart to ``export_text`` — lets a teammate who only received
    the JSONL file (e.g. over git/Slack, without access to this machine's
    Elasticsearch volume) reconstruct the text index locally with one
    command, same as ``build-index`` reconstructs Qdrant from the local
    ``embeddings/*.npz`` files.
    """
    manifest_path = experiment.run_dir / "manifests" / "text.jsonl"
    if not manifest_path.exists():
        raise CodeNovaError(f"No text.jsonl found at {manifest_path}. Run export-text first.")

    text_index = build_text_index(experiment)
    documents = [
        TextDocument(
            doc_id=row["doc_id"],
            video_id=row["video_id"],
            text=row["text"],
            source=row["source"],
            frame_id=row.get("frame_id"),
            timestamp_sec=row.get("timestamp_sec"),
        )
        for row in JsonlManifest(manifest_path).read_all()
    ]
    text_index.index_documents(documents)
    LOGGER.info("Imported %s text documents from %s", len(documents), manifest_path)
    return len(documents)
