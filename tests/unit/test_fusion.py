from core.types import SearchResult
from retrieval.fusion import srrf_fuse


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

    fused = srrf_fuse(
        {"m1": list_1, "m2": list_2}, top_k=10, weights={"m1": 2.0, "m2": 0.5}
    )

    assert fused[0].frame_id == "a"
