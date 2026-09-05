import json

from config.settings import Experiment, PipelineConfig
from core.types import FrameRecord, ShotRecord
from indexing.build_index import build_index
from indexing.embeddings import embed_frames
from indexing.frames import extract_frames
from indexing.ingest import ingest_videos
from indexing.manifest import JsonlManifest
from indexing.readiness import write_readiness
from indexing.shots import detect_shots
from indexing.validation import validate_experiment_artifacts, verify_artifact_fingerprints


class FakeDetector:
    def __init__(self, **kwargs):
        pass

    def detect_decoded(self, video, decoded):
        return [ShotRecord(video.video_id, f"{video.video_id}_s1", 0, 10)]


class FakeEmbedder:
    def __init__(self, **kwargs):
        pass

    def embed_images(self, frames, on_batch=None):
        vectors = [[float(index), 1.0] for index, _ in enumerate(frames, start=1)]
        if on_batch:
            on_batch(frames, vectors)
        return vectors


class FakeVectorIndex:
    def __init__(self):
        self.frame_ids = []

    def build(self, embeddings_by_model, frame_ids):
        self.frame_ids = list(frame_ids)


def test_fake_offline_pipeline_reaches_fresh_readiness(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_bytes(b"fake-video")
    experiment = Experiment(
        "exp",
        tmp_path / "run",
        PipelineConfig(runs_dir=tmp_path, embedding_models=("jina-clip-v2",)),
    )

    assert ingest_videos(experiment, input_dir) == 1
    video_id = next(
        iter(
            JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl").ids(
                "video_id", strict=True
            )
        )
    )

    monkeypatch.setattr("indexing.shots.TransNetV2ShotDetector", FakeDetector)
    monkeypatch.setattr("indexing.shots.decode_video", lambda path: object())
    assert detect_shots(experiment, tmp_path / "weights") == 1

    def fake_extract(self, video, shots):
        path = self.output_dir / video.video_id / f"{video.video_id}_s1_f5.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"frame")
        return [
            FrameRecord(
                f"{video.video_id}_s1_f5",
                video.video_id,
                f"{video.video_id}_s1",
                str(path),
                5,
                0.2,
            )
        ]

    monkeypatch.setattr("indexing.frames.FFmpegFrameExtractor.extract", fake_extract)
    assert extract_frames(experiment) == 1

    monkeypatch.setattr("indexing.embeddings.build_embedder", lambda **kwargs: FakeEmbedder())
    assert embed_frames(experiment) == 1

    vector_index = FakeVectorIndex()
    monkeypatch.setattr("indexing.build_index.build_vector_index", lambda experiment: vector_index)
    assert build_index(experiment) == 1
    assert vector_index.frame_ids == [f"{video_id}_s1_f5"]

    report = validate_experiment_artifacts(experiment)
    assert report.status == "READY", [issue.to_dict() for issue in report.issues]
    readiness = write_readiness(experiment, report)
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    assert verify_artifact_fingerprints(payload) == []
