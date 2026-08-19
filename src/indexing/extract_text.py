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

    Because the rebuild only reads from partitions, a ``text.jsonl`` written
    by anything that predates the partition scheme (or whose rows never made
    it into a partition for any other reason) would silently vanish the
    first time a new document triggers a rewrite. ``__init__`` guards against
    that: on first use, any ``text.jsonl`` row not yet backed by a partition
    file is migrated in before this sink accepts its first write.
    """

    def __init__(self, experiment) -> None:
        self._index = build_text_index(experiment)
        self._manifest = JsonlManifest(experiment.run_dir / "manifests" / "text.jsonl")
        self._partition_root = experiment.run_dir / "manifests" / "partitions" / "text"
        self._lock = threading.Lock()
        self._migrate_legacy_manifest()

    def _migrate_legacy_manifest(self) -> None:
        """One-time backfill: partition any text.jsonl rows not already partitioned."""
        if not self._manifest.path.exists():
            return
        existing_doc_ids: set[str] = set()
        for path in self._partition_root.glob("*/*.jsonl"):
            existing_doc_ids.update(
                str(row["doc_id"]) for row in JsonlManifest(path).read_all(strict=True)
            )

        orphaned = [
            row
            for row in self._manifest.read_all(strict=True)
            if str(row.get("doc_id")) not in existing_doc_ids
        ]
        if not orphaned:
            return

        LOGGER.warning(
            "Migrating %s pre-partition text.jsonl row(s) into partitions "
            "(source=%s) before accepting new writes",
            len(orphaned),
            self._partition_root,
        )
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in orphaned:
            key = (str(row.get("source")), str(row.get("video_id")))
            grouped.setdefault(key, []).append(row)

        for (source, video_id), rows in grouped.items():
            store = PartitionStore(self._partition_root / source)
            existing_path = store.path_for(video_id)
            by_id = (
                {
                    str(existing_row["doc_id"]): existing_row
                    for existing_row in JsonlManifest(existing_path).read_all(strict=True)
                }
                if existing_path.exists()
                else {}
            )
            by_id.update({str(row["doc_id"]): row for row in rows})
            store.write(video_id, by_id.values(), partition_key="video_id", unique_key="doc_id")

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
            # backup=False: this runs once per video (hundreds of times per
            # extract-text run), and text.jsonl only grows larger as the run
            # progresses. backup=True copied the whole file — tens to
            # hundreds of MB — on every single call, unbounded, and none of
            # those .bak files were ever cleaned up; a long run has filled
            # the disk with them before (~13GB from one session). The
            # per-video partitions under manifests/partitions/text/ plus
            # Elasticsearch are the durable copies here, so a rewrite
            # snapshot isn't needed on every write.
            self._manifest.replace_all(rows, unique_key="doc_id", backup=False)
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
    """Re-dump every text document from Elasticsearch to ``manifests/text.jsonl``.

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


def import_text(experiment, *, include_captions: bool = True) -> int:
    """Load local text manifests back into Elasticsearch.

    The counterpart to ``export_text`` — lets a teammate who only received
    the JSONL file (e.g. over git/Slack, without access to this machine's
    Elasticsearch volume) reconstruct the text index locally with one
    command, same as ``build-index`` reconstructs Qdrant from the local
    ``embeddings/*.npz`` files. Caption documents are included by default
    when ``manifests/captions.jsonl`` exists. Elasticsearch uses the stable
    ``doc_id`` as ``_id``, so repeated imports replace the same records rather
    than creating duplicates.
    """
    manifest_path = experiment.run_dir / "manifests" / "text.jsonl"
    caption_path = experiment.run_dir / "manifests" / "captions.jsonl"
    if not manifest_path.exists() and not (include_captions and caption_path.exists()):
        raise CodeNovaError(
            f"No importable text manifest found at {manifest_path}"
            + (" or " + str(caption_path) if include_captions else "")
            + ". Run extract-text/export-text first."
        )

    from modules.captioning.validation import validate_caption

    documents: list[TextDocument] = []
    skipped_caption_rows = 0
    manifest_rows = JsonlManifest(manifest_path).read_all() if manifest_path.exists() else []
    for row in manifest_rows:
        source = row["source"]
        text = row["text"]
        if source == "caption":
            if not include_captions:
                continue
            if (
                not isinstance(text, str)
                or not text.strip()
                or not validate_caption(text).valid
            ):
                skipped_caption_rows += 1
                continue
        documents.append(
            TextDocument(
                doc_id=row["doc_id"],
                video_id=row["video_id"],
                text=text.strip() if source == "caption" and isinstance(text, str) else text,
                source=source,
                frame_id=row.get("frame_id"),
                timestamp_sec=row.get("timestamp_sec"),
            )
        )
    if skipped_caption_rows:
        LOGGER.warning(
            "Skipped %s empty or invalid caption row(s) from %s",
            skipped_caption_rows,
            manifest_path,
        )

    if include_captions:
        if caption_path.exists():
            # A full Elasticsearch export may already contain caption rows.
            # Keying the combined batch by doc_id keeps the import count and
            # bulk request idempotent in that case as well.
            by_id = {document.doc_id: document for document in documents}
            for document in _caption_documents(experiment):
                by_id[document.doc_id] = document
            documents = list(by_id.values())

    text_index = build_text_index(experiment)
    text_index.index_documents(documents)
    LOGGER.info(
        "Imported %s text documents from %s (captions=%s)",
        len(documents),
        manifest_path,
        "included" if include_captions else "disabled",
    )
    return len(documents)


def import_captions(experiment) -> int:
    """Load ``manifests/captions.jsonl`` into Elasticsearch for intelligent search."""
    documents = _caption_documents(experiment)

    if documents:
        build_text_index(experiment).index_documents(documents)
    LOGGER.info("Imported %s caption documents", len(documents))
    return len(documents)


def _caption_documents(experiment) -> list[TextDocument]:
    """Build valid, frame-linked caption documents without mutating an index."""
    from modules.captioning.validation import validate_caption

    captions_path = experiment.run_dir / "manifests" / "captions.jsonl"
    if not captions_path.exists():
        return []

    frames_manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    frame_records = {
        row["frame_id"]: FrameRecord.from_dict(row) for row in frames_manifest.read_all()
    }

    documents_by_id: dict[str, TextDocument] = {}
    skipped = 0
    for row in JsonlManifest(captions_path).read_all():
        frame_id = row.get("frame_id")
        caption = row.get("caption")
        if not isinstance(frame_id, str) or not frame_id:
            skipped += 1
            continue
        if not isinstance(caption, str) or not caption.strip():
            skipped += 1
            continue
        if not validate_caption(caption).valid:
            skipped += 1
            continue
        record = frame_records.get(frame_id)
        if record is None:
            skipped += 1
            continue
        document = TextDocument(
            doc_id=f"{frame_id}__caption",
            video_id=record.video_id,
            text=caption.strip(),
            source="caption",
            frame_id=frame_id,
            timestamp_sec=record.timestamp_sec,
        )
        documents_by_id[document.doc_id] = document

    if skipped:
        LOGGER.warning("Skipped %s empty, invalid, or unlinked caption record(s)", skipped)
    return list(documents_by_id.values())
