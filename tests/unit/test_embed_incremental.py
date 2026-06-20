import json

from config.settings import Experiment, PipelineConfig
from core.types import FrameRecord
from indexing import embeddings
from indexing.manifest import JsonlManifest


class FakeEmbedder:
    """Embeds each frame as a 2-D vector derived from its frame_id length."""

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[list[str]] = []

    def embed_images(self, frames: list[FrameRecord]) -> list[list[float]]:
        self.calls.append([f.frame_id for f in frames])
        return [[float(len(f.frame_id)), 1.0] for f in frames]


def _experiment(tmp_path) -> Experiment:
    return Experiment(name="exp", run_dir=tmp_path, config=PipelineConfig(runs_dir=tmp_path))


def _add_frame(tmp_path, frame_id: str) -> None:
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").append(
        {
            "frame_id": frame_id,
            "video_id": frame_id.split("_")[0],
            "shot_id": frame_id.rsplit("_", 1)[0],
            "frame_path": f"/x/{frame_id}.jpg",
        }
    )


def test_incremental_embeds_only_new_frames(tmp_path, monkeypatch) -> None:
    import numpy as np

    exp = _experiment(tmp_path)
    fake = FakeEmbedder()
    monkeypatch.setattr(embeddings, "build_embedder", lambda **kw: fake)

    _add_frame(tmp_path, "v1_s0_f1")
    _add_frame(tmp_path, "v1_s0_f2")
    added = embeddings.embed_frames(exp)
    assert added == 2

    # add one more frame, re-run: only the new one is embedded
    _add_frame(tmp_path, "v1_s0_f3")
    added = embeddings.embed_frames(exp)
    assert added == 1
    assert fake.calls[-1] == ["v1_s0_f3"]  # only the new frame was embedded

    # npz rows and frame_ids stay aligned and complete
    ids = json.loads((tmp_path / "embeddings" / "frame_ids.json").read_text())
    vecs = np.load(tmp_path / "embeddings" / "frames.npz")["embeddings"]
    assert ids == ["v1_s0_f1", "v1_s0_f2", "v1_s0_f3"]
    assert vecs.shape == (3, 2)


def test_rerun_with_no_new_frames_is_noop(tmp_path, monkeypatch) -> None:
    exp = _experiment(tmp_path)
    fake = FakeEmbedder()
    monkeypatch.setattr(embeddings, "build_embedder", lambda **kw: fake)

    _add_frame(tmp_path, "v1_s0_f1")
    embeddings.embed_frames(exp)
    assert embeddings.embed_frames(exp) == 0
