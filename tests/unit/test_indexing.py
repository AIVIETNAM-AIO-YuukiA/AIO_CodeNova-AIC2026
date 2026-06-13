from config.settings import Experiment, PipelineConfig
from core.types import SearchResult
from pipeline.manifest import JsonlManifest
from pipeline import indexing


class FakeEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def embed_text(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeIndex:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        return [SearchResult(frame_id="video1_s000001_f00000042", video_id="", score=0.9)]


def test_search_index_hydrates_video_filename_and_frame_index(tmp_path, monkeypatch) -> None:
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
            "frame_path": "runs/exp/frames/video1/video1_s000001_f00000042.jpg",
            "frame_index": 42,
            "timestamp_sec": 1.4,
        }
    )
    monkeypatch.setattr(indexing, "TransformersClipEmbedder", FakeEmbedder)
    monkeypatch.setattr(indexing, "FaissVectorIndex", FakeIndex)

    results = indexing.search_index(experiment, query="sample", top_k=1)

    assert results[0].video_name == "sample.mp4"
    assert results[0].video_path == "data/raw_videos/sample.mp4"
    assert results[0].frame_index == 42
    assert results[0].shot_id == "video1_s000001"
