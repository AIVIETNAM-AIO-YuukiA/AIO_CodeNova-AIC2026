"""OCR + ASR text extraction stage — offline, indexes into Elasticsearch.

Two independent sub-stages sharing one job-state table (``EXTRACT_OCR`` /
``EXTRACT_ASR``), each resumable on its own:

- OCR runs per-keyframe (``modules/ocr/vllm.py``, same self-hosted VLM as
  captioning, dedicated OCR-only prompt) — on-screen ticker/banner/program
  name/logo text, the single most query-able detail in a news keyframe.
- ASR runs per-video (``modules/asr/gipformer.py``) — the video's full audio
  track, chunked and transcribed.

Both write ``TextDocument`` records into the configured ``TextIndex``
(Elasticsearch) via ``stores/text/factory.py``, keyed by the same
``frame_id``/``video_id`` used everywhere else in the pipeline, so results
here fuse naturally with vector search — this is the OCR/ASR branch the
project's docs have described as "not yet wired in" until now.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

from core.errors import CodeNovaError
from core.logging import get_logger
from core.types import FrameRecord, VideoRecord
from indexing.manifest import JsonlManifest
from indexing.state import JobState
from modules.asr.gipformer import GipformerAsrModel
from modules.ocr.vllm import VllmOcrModel
from stores.text.base import TextDocument
from stores.text.factory import build_text_index

LOGGER = get_logger(__name__)

_DEFAULT_OCR_WORKERS = 8


def extract_ocr(experiment, force: bool = False) -> int:
    """Run OCR on every keyframe and index the results into Elasticsearch.

    Returns the number of frames newly indexed (frames with no on-screen text
    are still marked completed, so they aren't retried every run, but produce
    no document — an empty Elasticsearch document would just be noise).

    Runs concurrently across keyframes (CAPTION_WORKERS threads, shared with
    the captioning branch's default) since each call is a slow vLLM round
    trip — see VietnameseEmbedder.embed_images for the same rationale.
    JobState.mark and the Elasticsearch client are each safe to call from
    multiple threads (SQLite serializes writes per-connection; the ES client
    is a standard pooled HTTP client), so no extra locking is needed here.
    """
    frames_manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    frames = [FrameRecord.from_dict(row) for row in frames_manifest.read_all()]
    frames = [f for f in frames if not state.should_skip(f.frame_id, "EXTRACT_OCR", force=force)]
    if not frames:
        LOGGER.warning("No frames to OCR (all done, or none found)")
        return 0

    ocr_model = VllmOcrModel()
    text_index = build_text_index(experiment)
    workers = int(os.environ.get("CAPTION_WORKERS", _DEFAULT_OCR_WORKERS))

    def _process(frame: FrameRecord) -> bool:
        try:
            text = ocr_model.recognize(frame.frame_path)
            if text:
                text_index.index_documents(
                    [
                        TextDocument(
                            doc_id=f"{frame.frame_id}__ocr",
                            video_id=frame.video_id,
                            frame_id=frame.frame_id,
                            text=text,
                            source="ocr",
                            timestamp_sec=frame.timestamp_sec,
                        )
                    ]
                )
            state.mark(frame.frame_id, "EXTRACT_OCR", "COMPLETED")
            return bool(text)
        except Exception as exc:
            LOGGER.exception("OCR failed frame_id=%s", frame.frame_id)
            state.mark(frame.frame_id, "EXTRACT_OCR", "FAILED", error=str(exc))
            return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process, frames))

    indexed = sum(results)
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
    text_index = build_text_index(experiment)
    indexed = 0

    for video in videos:
        if state.should_skip(video.video_id, "EXTRACT_ASR", force=force):
            continue
        try:
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
                text_index.index_documents(documents)
                indexed += len(documents)
            state.mark(video.video_id, "EXTRACT_ASR", "COMPLETED")
        except Exception as exc:
            LOGGER.exception("ASR failed video_id=%s", video.video_id)
            state.mark(video.video_id, "EXTRACT_ASR", "FAILED", error=str(exc))

    LOGGER.info("ASR extraction complete: %s segments indexed", indexed)
    return indexed


def export_text(experiment) -> int:
    """Dump every OCR/ASR document from Elasticsearch to ``manifests/text.jsonl``.

    Elasticsearch (a Docker-managed volume, not a project directory) is the
    system of record for OCR/ASR text — this just makes a local JSONL copy
    for backup/inspection without needing to query Elasticsearch directly,
    mirroring how ``captions.jsonl`` mirrors the Vietnamese embedder's captions.
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
