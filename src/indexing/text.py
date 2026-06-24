"""OCR/ASR text extraction and text-index build stages."""

from __future__ import annotations

from core.logging import get_logger
from core.types import FrameRecord, VideoRecord
from config.settings import Experiment
from indexing.manifest import JsonlManifest
from indexing.state import JobState
from modules.asr import AsrModel, build_asr_model
from modules.ocr import OcrModel, build_ocr_model
from stores.text import TextDocument, build_text_index

LOGGER = get_logger(__name__)


def run_ocr(experiment: Experiment, force: bool = False, model: OcrModel | None = None) -> int:
    """Recognize text in extracted keyframes and persist OCR artifacts."""
    frames_manifest = JsonlManifest(experiment.run_dir / "manifests" / "frames.jsonl")
    ocr_manifest = JsonlManifest(experiment.run_dir / "text" / "ocr.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    ocr_model = model or build_ocr_model()
    existing_ids = ocr_manifest.ids("frame_id")
    recorded = 0

    for row in frames_manifest.read_all():
        frame = FrameRecord.from_dict(row)
        if state.should_skip(frame.frame_id, "OCR", force=force) and frame.frame_id in existing_ids:
            LOGGER.info("Skipping OCR frame_id=%s", frame.frame_id)
            continue
        try:
            text = ocr_model.recognize(frame.frame_path)
            ocr_manifest.append(
                {
                    "frame_id": frame.frame_id,
                    "video_id": frame.video_id,
                    "shot_id": frame.shot_id,
                    "frame_path": frame.frame_path,
                    "timestamp_sec": frame.timestamp_sec,
                    "text": text,
                }
            )
            state.mark(frame.frame_id, "OCR", "COMPLETED")
            recorded += 1
            LOGGER.info("Recorded OCR frame_id=%s chars=%s", frame.frame_id, len(text))
        except Exception as exc:
            LOGGER.exception("OCR failed frame_id=%s", frame.frame_id)
            state.mark(frame.frame_id, "OCR", "FAILED", error=str(exc))
    return recorded


def run_asr(experiment: Experiment, force: bool = False, model: AsrModel | None = None) -> int:
    """Transcribe discovered videos and persist ASR artifacts."""
    videos_manifest = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl")
    asr_manifest = JsonlManifest(experiment.run_dir / "text" / "asr.jsonl")
    state = JobState(experiment.run_dir / "jobs.sqlite")
    asr_model = model or build_asr_model()
    existing_ids = asr_manifest.ids("video_id")
    recorded = 0

    for row in videos_manifest.read_all():
        video = VideoRecord.from_dict(row)
        if state.should_skip(video.video_id, "ASR", force=force) and video.video_id in existing_ids:
            LOGGER.info("Skipping ASR video_id=%s", video.video_id)
            continue
        try:
            transcripts = asr_model.transcribe(video.path)
            asr_manifest.extend(
                {
                    "video_id": video.video_id,
                    "video_path": video.path,
                    "text": transcript.text,
                    "start_time_sec": transcript.start_time_sec,
                    "end_time_sec": transcript.end_time_sec,
                }
                for transcript in transcripts
            )
            state.mark(video.video_id, "ASR", "COMPLETED")
            recorded += len(transcripts)
            LOGGER.info("Recorded ASR video_id=%s segments=%s", video.video_id, len(transcripts))
        except Exception as exc:
            LOGGER.exception("ASR failed video_id=%s", video.video_id)
            state.mark(video.video_id, "ASR", "FAILED", error=str(exc))
    return recorded


def build_text_index_from_artifacts(experiment: Experiment, force: bool = False) -> int:
    """Index OCR and ASR artifacts into the configured text index."""
    del force
    documents = load_text_documents(experiment)
    index = build_text_index(experiment)
    index.index_documents(documents)
    return len(documents)


def load_text_documents(experiment: Experiment) -> list[TextDocument]:
    """Load OCR/ASR JSONL artifacts as indexable text documents."""
    documents: list[TextDocument] = []
    ocr_manifest = JsonlManifest(experiment.run_dir / "text" / "ocr.jsonl")
    for row in ocr_manifest.read_all():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        frame_id = str(row["frame_id"])
        documents.append(
            TextDocument(
                doc_id=f"ocr:{frame_id}",
                video_id=str(row["video_id"]),
                frame_id=frame_id,
                text=text,
                source="ocr",
                timestamp_sec=_optional_float(row.get("timestamp_sec")),
            )
        )

    asr_manifest = JsonlManifest(experiment.run_dir / "text" / "asr.jsonl")
    for ordinal, row in enumerate(asr_manifest.read_all()):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        video_id = str(row["video_id"])
        start = _optional_float(row.get("start_time_sec"))
        suffix = str(start) if start is not None else str(ordinal)
        documents.append(
            TextDocument(
                doc_id=f"asr:{video_id}:{suffix}",
                video_id=video_id,
                frame_id=None,
                text=text,
                source="asr",
                timestamp_sec=start,
            )
        )
    return documents


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
