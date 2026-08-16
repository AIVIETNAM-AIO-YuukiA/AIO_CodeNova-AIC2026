import pytest

from config.settings import Experiment, PipelineConfig
from core.types import FrameRecord, ShotRecord
from indexing.embeddings import embed_frames
from indexing.frames import extract_frames
from indexing.ingest import ingest_videos
from indexing.manifest import JsonlManifest, ManifestError
from indexing.shots import detect_shots
from indexing.state import JobState


def _experiment(tmp_path):
    return Experiment(
        name="exp",
        run_dir=tmp_path / "run",
        config=PipelineConfig(runs_dir=tmp_path),
    )


def test_ingest_consolidates_partitions_without_duplicate_records(tmp_path):
    experiment = _experiment(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.mp4").write_bytes(b"a")
    (input_dir / "b.mp4").write_bytes(b"b")

    assert ingest_videos(experiment, input_dir) == 2
    assert ingest_videos(experiment, input_dir) == 0

    rows = JsonlManifest(experiment.run_dir / "manifests" / "videos.jsonl").read_all(strict=True)
    assert len(rows) == 2
    assert len({row["video_id"] for row in rows}) == 2


def test_frame_stage_publishes_staging_files_and_manifest_partition(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    manifests = experiment.run_dir / "manifests"
    JsonlManifest(manifests / "videos.jsonl").append(
        {"video_id": "v1", "path": "v1.mp4", "checksum": "x", "size_bytes": 1}
    )
    JsonlManifest(manifests / "shots.jsonl").append(
        {"video_id": "v1", "shot_id": "v1_s1", "start_frame": 0, "end_frame": 10}
    )

    def fake_extract(self, video, shots):
        output = self.output_dir / video.video_id / "v1_s1_f00000005.jpg"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"frame")
        return [
            FrameRecord(
                frame_id="v1_s1_f00000005",
                video_id="v1",
                shot_id="v1_s1",
                frame_path=str(output),
                frame_index=5,
                timestamp_sec=0.2,
            )
        ]

    monkeypatch.setattr("indexing.frames.FFmpegFrameExtractor.extract", fake_extract)

    assert extract_frames(experiment) == 1
    rows = JsonlManifest(manifests / "frames.jsonl").read_all(strict=True)
    assert len(rows) == 1
    assert rows[0]["frame_path"] == "frames/v1/v1_s1_f00000005.jpg"
    assert (experiment.run_dir / str(rows[0]["frame_path"])).exists()
    assert ".staging" not in str(rows[0]["frame_path"])


def test_shot_stage_continues_after_one_video_fails(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    manifests = experiment.run_dir / "manifests"
    JsonlManifest(manifests / "videos.jsonl").extend(
        [
            {"video_id": "bad", "path": "bad.mp4", "checksum": "a", "size_bytes": 1},
            {"video_id": "good", "path": "good.mp4", "checksum": "b", "size_bytes": 1},
        ]
    )

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def detect_decoded(self, video, decoded):
            if video.video_id == "bad":
                raise RuntimeError("detector failed")
            return [ShotRecord(video.video_id, f"{video.video_id}_s1", 0, 10)]

    monkeypatch.setattr("indexing.shots.TransNetV2ShotDetector", FakeDetector)
    monkeypatch.setattr("indexing.shots.decode_video", lambda path: object())

    assert detect_shots(experiment, tmp_path / "weights") == 1
    rows = JsonlManifest(manifests / "shots.jsonl").read_all(strict=True)
    assert [row["video_id"] for row in rows] == ["good"]
    state = JobState(experiment.run_dir / "jobs.sqlite")
    assert state.get_status("bad", "SHOT_DETECT") == "FAILED"
    assert state.get_status("good", "SHOT_DETECT") == "COMPLETED"


def test_frame_stage_continues_after_failure_and_processes_next_video(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    manifests = experiment.run_dir / "manifests"
    JsonlManifest(manifests / "videos.jsonl").extend(
        [
            {"video_id": "bad", "path": "bad.mp4", "checksum": "a", "size_bytes": 1},
            {"video_id": "good", "path": "good.mp4", "checksum": "b", "size_bytes": 1},
        ]
    )
    JsonlManifest(manifests / "shots.jsonl").extend(
        [
            {"video_id": "bad", "shot_id": "bad_s1", "start_frame": 0, "end_frame": 1},
            {"video_id": "good", "shot_id": "good_s1", "start_frame": 0, "end_frame": 1},
        ]
    )

    def fake_extract(self, video, shots):
        if video.video_id == "bad":
            raise RuntimeError("ffmpeg failed")
        output = self.output_dir / video.video_id / "good_s1_f1.jpg"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"frame")
        return [FrameRecord("good_s1_f1", "good", "good_s1", str(output), 1, 0.04)]

    monkeypatch.setattr("indexing.frames.FFmpegFrameExtractor.extract", fake_extract)

    assert extract_frames(experiment) == 1
    rows = JsonlManifest(manifests / "frames.jsonl").read_all(strict=True)
    assert [row["video_id"] for row in rows] == ["good"]
    assert (
        JobState(experiment.run_dir / "jobs.sqlite").get_status("bad", "FRAME_EXTRACT") == "FAILED"
    )


def test_downstream_stages_reject_missing_prerequisite_manifests(tmp_path):
    experiment = _experiment(tmp_path)

    with pytest.raises(ManifestError, match="video manifest"):
        detect_shots(experiment, tmp_path / "weights")
    with pytest.raises(ManifestError, match="video manifest"):
        extract_frames(experiment)
    with pytest.raises(ManifestError, match="frame manifest"):
        embed_frames(experiment)
