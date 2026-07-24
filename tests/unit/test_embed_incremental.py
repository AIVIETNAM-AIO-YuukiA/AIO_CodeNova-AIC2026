import json

from config.settings import Experiment, PipelineConfig
from core.types import FrameRecord
from indexing import embeddings
from indexing.manifest import JsonlManifest


class FakeEmbedder:
    """Embeds each frame as a 2-D vector derived from its frame_id length.

    Calls ``on_batch`` per frame (batch size 1) so tests can exercise the
    checkpoint path the same way a real backend's internal batch loop would.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[list[str]] = []

    def embed_images(self, frames: list[FrameRecord], on_batch=None) -> list[list[float]]:
        self.calls.append([f.frame_id for f in frames])
        vectors = []
        for frame in frames:
            vector = [float(len(frame.frame_id)), 1.0]
            vectors.append(vector)
            if on_batch is not None:
                on_batch([frame], [vector])
        return vectors


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


def test_resumes_from_mid_model_checkpoint(tmp_path, monkeypatch) -> None:
    """Simulate a run killed mid-model: a checkpoint file exists (from a
    prior process that flushed partial progress) but frames.npz does not.
    Re-running must skip the checkpointed frames and only embed the rest."""
    import numpy as np

    exp = _experiment(tmp_path)
    fake = FakeEmbedder()
    monkeypatch.setattr(embeddings, "build_embedder", lambda **kw: fake)

    for frame_id in ["v1_s0_f1", "v1_s0_f2", "v1_s0_f3"]:
        _add_frame(tmp_path, frame_id)

    output_dir = tmp_path / "embeddings"
    output_dir.mkdir(parents=True)
    # Hand-write a checkpoint as if a prior interrupted pass had flushed
    # after embedding only the first frame.
    (output_dir / "frame_ids.checkpoint.json").write_text(json.dumps(["v1_s0_f1"]))
    np.savez_compressed(output_dir / "frames.checkpoint.npz", embeddings=np.array([[8.0, 1.0]]))

    added = embeddings.embed_frames(exp)

    assert added == 3  # 1 carried over from checkpoint + 2 freshly embedded
    assert fake.calls == [["v1_s0_f2", "v1_s0_f3"]]  # checkpointed frame skipped

    ids = json.loads((output_dir / "frame_ids.json").read_text())
    vecs = np.load(output_dir / "frames.npz")["embeddings"]
    assert ids == ["v1_s0_f1", "v1_s0_f2", "v1_s0_f3"]
    assert vecs.shape == (3, 2)

    # checkpoint files are cleaned up once the model finishes
    assert not (output_dir / "frame_ids.checkpoint.json").exists()
    assert not (output_dir / "frames.checkpoint.npz").exists()
