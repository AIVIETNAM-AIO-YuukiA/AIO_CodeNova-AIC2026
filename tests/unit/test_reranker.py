from unittest.mock import patch, MagicMock

from core.types import SearchResult
from modules.reranker.base import build_reranker
from modules.reranker.blip2_itm import Blip2ItmReranker
from modules.reranker.qwen_vl_vllm import QwenVlVllmReranker


def test_build_reranker():
    """Test the reranker factory function and aliases."""
    # None should return None
    assert build_reranker(None) is None

    # Valid alias should resolve and return instance
    reranker = build_reranker("blip2-itm", device="cpu", batch_size=4)
    assert isinstance(reranker, Blip2ItmReranker)
    assert reranker.model_name == "Salesforce/blip2-itm-vit-g"
    assert reranker.device == "cpu"
    assert reranker.batch_size == 4


def test_build_reranker_qwen_vl_vllm_backend(monkeypatch):
    """RERANKER_BACKEND=qwen-vl-vllm routes to the vLLM-backed implementation."""
    monkeypatch.setenv("RERANKER_BACKEND", "qwen-vl-vllm")
    monkeypatch.setenv("QWEN_VL_RERANKER_URL", "http://example:8884")
    monkeypatch.setenv("QWEN_VL_RERANKER_MODEL", "Qwen/Qwen3-VL-Reranker-2B")

    reranker = build_reranker("blip2-itm", device="cpu", batch_size=4)

    assert isinstance(reranker, QwenVlVllmReranker)
    assert reranker.base_url == "http://example:8884"
    assert reranker.model_name == "Qwen/Qwen3-VL-Reranker-2B"
    assert reranker.batch_size == 4


def test_lazy_loading():
    """Ensure the model is not loaded during initialization."""
    reranker = Blip2ItmReranker(model_name="dummy", device="cpu")
    assert reranker._model is None
    assert reranker._processor is None

    # Reranking an empty list should still not load the model
    result = reranker.rerank("A test query", [])
    assert result == []
    assert reranker._model is None


@patch("modules.reranker.blip2_itm.Blip2ItmReranker._load")
@patch("modules.reranker.blip2_itm.Blip2ItmReranker._score_batch")
def test_rerank_logic(mock_score_batch, mock_load, tmp_path):
    """Test that missing frames are dropped and valid frames are scored and sorted."""

    # Mock _load to return dummy model/processor
    mock_load.return_value = (MagicMock(), MagicMock(), MagicMock(), "cpu")

    # Mock _score_batch to return fake scores [0.1, 0.9] for the valid frames
    mock_score_batch.return_value = [0.1, 0.9]

    reranker = Blip2ItmReranker(model_name="dummy", device="cpu")

    # Create valid frame files
    valid_frame1 = tmp_path / "valid1.jpg"
    valid_frame1.touch()
    valid_frame2 = tmp_path / "valid2.jpg"
    valid_frame2.touch()

    # Create SearchResult objects (one has a missing frame)
    results = [
        SearchResult(
            video_id="v1",
            video_name="vid1",
            frame_id="f1",
            frame_index=10,
            timestamp_sec=1.0,
            score=0.8,
            frame_path=str(valid_frame1),
        ),
        SearchResult(
            video_id="v2",
            video_name="vid2",
            frame_id="f2",
            frame_index=20,
            timestamp_sec=2.0,
            score=0.85,
            frame_path="missing_frame.jpg",
        ),
        SearchResult(
            video_id="v3",
            video_name="vid3",
            frame_id="f3",
            frame_index=30,
            timestamp_sec=3.0,
            score=0.7,
            frame_path=str(valid_frame2),
        ),
    ]

    # Rerank
    query = "test query"
    reranked = reranker.rerank(query, results)

    # Assertions
    assert mock_load.called
    assert mock_score_batch.called

    # Should only return the 2 valid results
    assert len(reranked) == 2

    # rerank() blends ITM score with normalized CLIP score (hybrid formula,
    # itm_weight=0.6 default) rather than returning the raw ITM score — see
    # Blip2ItmReranker's docstring. valid_results are [vid1 (clip=0.8), vid3
    # (clip=0.7)], itm_scores are [0.1, 0.9] in that order, so:
    #   vid3: clip_norm=0.0 (min of the pool) -> hybrid = 0.6*0.9 + 0.4*0 = 0.54
    #   vid1: clip_norm=1.0 (max of the pool) -> hybrid = 0.6*0.1 + 0.4*1 = 0.46
    assert reranked[0].video_name == "vid3"
    assert reranked[0].score == 0.54

    assert reranked[1].video_name == "vid1"
    assert reranked[1].score == 0.46
