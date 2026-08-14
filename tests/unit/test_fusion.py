import numpy as np
import pytest

from core.errors import FusionError
from core.types import SearchResult
from retrieval.fusion import srrf_fuse, wsf_fuse


def _result(frame_id: str, score: float) -> SearchResult:
    return SearchResult(frame_id=frame_id, video_id="v", score=score)


def test_single_list_passthrough() -> None:
    results = [_result("a", 0.9), _result("b", 0.5)]

    fused = srrf_fuse({"m1": results}, top_k=10)

    assert fused == results[:10]


def test_agreement_across_lists_ranks_first() -> None:
    # "a" is top-1 in both lists; "b" and "c" only appear in one list each.
    list_1 = [_result("a", 0.95), _result("b", 0.40)]
    list_2 = [_result("a", 0.90), _result("c", 0.35)]

    fused = srrf_fuse({"m1": list_1, "m2": list_2}, top_k=10)

    assert fused[0].frame_id == "a"
    assert {r.frame_id for r in fused} == {"a", "b", "c"}


def test_top_k_is_respected() -> None:
    list_1 = [_result(f"f{i}", 1.0 - i * 0.01) for i in range(20)]
    list_2 = [_result(f"f{i}", 1.0 - i * 0.02) for i in range(20)]

    fused = srrf_fuse({"m1": list_1, "m2": list_2}, top_k=5)

    assert len(fused) == 5


def test_fused_score_is_not_raw_similarity() -> None:
    list_1 = [_result("a", 0.9)]
    list_2 = [_result("a", 0.8)]

    fused = srrf_fuse({"m1": list_1, "m2": list_2}, top_k=10)

    assert fused[0].score != 0.9
    assert fused[0].score != 0.8
    assert fused[0].score > 0


def test_weights_favor_higher_weighted_model() -> None:
    # "a" only appears in list_1 (heavily weighted); "b" only in list_2.
    list_1 = [_result("a", 0.9)]
    list_2 = [_result("b", 0.9)]

    fused = srrf_fuse({"m1": list_1, "m2": list_2}, top_k=10, weights={"m1": 2.0, "m2": 0.5})

    assert fused[0].frame_id == "a"


def _wsf_data(ids, vectors, query=(1.0, 0.0)):
    return (
        np.asarray(vectors, dtype="float32"),
        [{"frame_id": frame_id} for frame_id in ids],
        np.asarray(query, dtype="float32"),
    )


def test_wsf_aligns_reordered_models_by_frame_id() -> None:
    fused = wsf_fuse(
        {
            "m1": _wsf_data(["f1", "f2"], [[1.0, 0.0], [0.0, 1.0]]),
            "m2": _wsf_data(["f2", "f1"], [[0.0, 1.0], [1.0, 0.0]]),
        },
        top_k=2,
        weights={"m1": 0.1, "m2": 0.9},
    )

    assert [result.frame_id for result in fused] == ["f1", "f2"]


def test_wsf_rejects_different_frame_id_sets() -> None:
    with pytest.raises(FusionError, match="frame-ID set differs") as captured:
        wsf_fuse(
            {
                "m1": _wsf_data(["f1", "f2"], [[1.0, 0.0], [0.0, 1.0]]),
                "m2": _wsf_data(["f1"], [[1.0, 0.0]]),
            },
            top_k=2,
        )

    assert "missing=['f2']" in str(captured.value)


def test_wsf_rejects_duplicate_frame_ids() -> None:
    with pytest.raises(FusionError, match="duplicate frame_id"):
        wsf_fuse(
            {"m1": _wsf_data(["f1", "f1"], [[1.0, 0.0], [0.0, 1.0]])},
            top_k=2,
        )


def test_wsf_rejects_vector_record_count_mismatch() -> None:
    with pytest.raises(FusionError, match="vectors=2 records=1"):
        wsf_fuse(
            {"m1": _wsf_data(["f1"], [[1.0, 0.0], [0.0, 1.0]])},
            top_k=2,
        )
