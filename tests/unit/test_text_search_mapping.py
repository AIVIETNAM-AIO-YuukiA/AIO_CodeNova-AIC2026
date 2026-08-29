import math

import pytest

from config.settings import Experiment, PipelineConfig
from indexing.manifest import JsonlManifest
from retrieval.text_search import (
    AsrTemporalMapper,
    NearestFrameIndex,
    infer_asr_intervals,
)


def test_infer_asr_intervals_uses_next_start_and_caps_long_segments():
    records = [
        {"doc_id": "v1-3", "video_id": "v1", "source": "asr", "timestamp_sec": 80},
        {"doc_id": "v2-1", "video_id": "v2", "source": "asr", "timestamp_sec": 7},
        {"doc_id": "v1-1", "video_id": "v1", "source": "asr", "timestamp_sec": 0},
        {"doc_id": "v1-2", "video_id": "v1", "source": "asr", "timestamp_sec": 20},
        {"doc_id": "ocr", "video_id": "v1", "source": "ocr", "timestamp_sec": 30},
        {"doc_id": "bad", "video_id": "v1", "source": "asr", "timestamp_sec": None},
    ]

    intervals = infer_asr_intervals(records, max_duration_sec=45.0)

    assert intervals == {
        "v1-1": (0.0, 20.0),
        "v1-2": (20.0, 65.0),
        "v1-3": (80.0, 125.0),
        "v2-1": (7.0, 52.0),
    }


def test_nearest_frame_index_maps_interval_with_padding_decay_and_fallback():
    index = NearestFrameIndex.__new__(NearestFrameIndex)
    index._by_video = {"v1": [(0.0, "f0"), (3.0, "f3"), (5.0, "f5"), (8.0, "f8"), (20.0, "f20")]}

    weighted = dict(index.map_interval("v1", 5.0, 8.0, padding_sec=2.0, decay_sec=2.0))

    assert weighted["f5"] == 1.0
    assert weighted["f8"] == 1.0
    assert weighted["f3"] == pytest.approx(math.exp(-1.0))
    assert "f0" not in weighted
    assert index.map_interval("v1", 100.0, 105.0) == [("f20", 1.0)]


def test_nearest_frame_index_keeps_point_window_helpers():
    index = NearestFrameIndex.__new__(NearestFrameIndex)
    index._by_video = {"v1": [(0.0, "f0"), (1.0, "f1"), (3.0, "f3")]}

    nearby = index.within("v1", 1.0, 1.1)
    weighted = dict(index.nearby_weighted("v1", 1.0, 2.0))

    assert {frame_id for frame_id, _ in nearby} == {"f0", "f1"}
    assert weighted["f1"] == 1.0
    assert 0.0 < weighted["f0"] < 1.0


def test_asr_temporal_mapper_reads_legacy_manifest_without_rewriting_it(tmp_path):
    experiment = Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").extend(
        [
            {
                "frame_id": "f4",
                "video_id": "v1",
                "shot_id": "s1",
                "frame_path": "frames/v1/f4.jpg",
                "frame_index": 100,
                "timestamp_sec": 4.0,
            },
            {
                "frame_id": "f6",
                "video_id": "v1",
                "shot_id": "s1",
                "frame_path": "frames/v1/f6.jpg",
                "frame_index": 150,
                "timestamp_sec": 6.0,
            },
            {
                "frame_id": "f10",
                "video_id": "v1",
                "shot_id": "s2",
                "frame_path": "frames/v1/f10.jpg",
                "frame_index": 250,
                "timestamp_sec": 10.0,
            },
        ]
    )
    text_manifest = JsonlManifest(tmp_path / "manifests" / "text.jsonl")
    text_manifest.extend(
        [
            {
                "doc_id": "a1",
                "video_id": "v1",
                "source": "asr",
                "text": "xin chào",
                "timestamp_sec": 5.0,
            },
            {
                "doc_id": "a2",
                "video_id": "v1",
                "source": "asr",
                "text": "tin tiếp theo",
                "timestamp_sec": 10.0,
            },
        ]
    )
    before = text_manifest.path.read_text(encoding="utf-8")

    mapper = AsrTemporalMapper(experiment)
    mapped = dict(
        mapper.map_document(
            {"doc_id": "a1", "video_id": "v1", "source": "asr", "timestamp_sec": 5.0}
        )
    )

    assert mapper.interval_for({"doc_id": "a1"}) == (5.0, 10.0)
    assert mapped["f4"] == pytest.approx(math.exp(-0.5))
    assert mapped["f6"] == 1.0
    assert mapped["f10"] == 1.0
    assert text_manifest.path.read_text(encoding="utf-8") == before


def test_asr_temporal_mapper_caches_large_manifest_by_mtime(tmp_path, monkeypatch):
    experiment = Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))
    JsonlManifest(tmp_path / "manifests" / "frames.jsonl").append(
        {
            "frame_id": "f1",
            "video_id": "v1",
            "shot_id": "s1",
            "frame_path": "frames/v1/f1.jpg",
            "frame_index": 25,
            "timestamp_sec": 1.0,
        }
    )
    JsonlManifest(tmp_path / "manifests" / "text.jsonl").append(
        {
            "doc_id": "a1",
            "video_id": "v1",
            "source": "asr",
            "text": "xin chào",
            "timestamp_sec": 1.0,
        }
    )
    AsrTemporalMapper._interval_cache.clear()
    original = JsonlManifest.read_all
    text_reads = 0

    def counted_read(manifest, *args, **kwargs):
        nonlocal text_reads
        if manifest.path.name == "text.jsonl":
            text_reads += 1
        return original(manifest, *args, **kwargs)

    monkeypatch.setattr(JsonlManifest, "read_all", counted_read)

    AsrTemporalMapper(experiment)
    AsrTemporalMapper(experiment)

    assert text_reads == 1
