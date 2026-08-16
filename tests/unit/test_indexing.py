from config.settings import Experiment, PipelineConfig
from core.types import SearchResult
from indexing.manifest import JsonlManifest
from retrieval.hydrator import ResultHydrator


def test_hydrator_attaches_video_filename_and_frame_index(tmp_path) -> None:
    frame_path = tmp_path / "frames" / "video1" / "video1_s000001_f00000042.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"frame")
    experiment = Experiment(
        name="exp",
        run_dir=tmp_path,
        config=PipelineConfig(runs_dir=tmp_path),
    )
    JsonlManifest(tmp_path / "manifests" / "videos.jsonl").append(
        {
            "video_id": "video1",
            "path": "data/raw_videos/sample.mp4",
            "checksum": "abc",
            "size_bytes": 123,
        }
    )
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").append(
        {
            "frame_id": "video1_s000001_f00000042",
            "video_id": "video1",
            "shot_id": "video1_s000001",
            "frame_path": "frames/video1/video1_s000001_f00000042.jpg",
            "frame_index": 42,
            "timestamp_sec": 1.4,
        }
    )

    raw = [SearchResult(frame_id="video1_s000001_f00000042", video_id="", score=0.9)]
    results = ResultHydrator(experiment).hydrate(raw)

    assert results[0].video_name == "sample.mp4"
    assert results[0].video_path == "data/raw_videos/sample.mp4"
    assert results[0].frame_index == 42
    assert results[0].shot_id == "video1_s000001"


def test_hydrator_reports_and_drops_unknown_frame(tmp_path) -> None:
    experiment = Experiment(
        name="exp",
        run_dir=tmp_path,
        config=PipelineConfig(runs_dir=tmp_path),
    )
    raw = [SearchResult(frame_id="missing", video_id="", score=0.5)]

    batch = ResultHydrator(experiment).hydrate_with_diagnostics(raw)

    assert batch.results == []
    assert [(issue.frame_id, issue.reason) for issue in batch.issues] == [
        ("missing", "FRAME_METADATA_MISSING")
    ]
