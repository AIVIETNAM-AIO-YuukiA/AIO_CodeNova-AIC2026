"""Tests for VQA pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config.settings import Experiment
from core.types import SearchResult


@pytest.fixture
def mock_experiment(tmp_path: Path) -> Experiment:
    run_dir = tmp_path / "runs" / "test"
    run_dir.mkdir(parents=True)
    config = MagicMock()
    config.embedding_models = ("siglip2-large",)
    config.device = "cpu"
    config.data_dir = tmp_path / "data"
    return Experiment(name="test", run_dir=run_dir, config=config)


@pytest.fixture(autouse=True)
def clear_retriever_cache():
    """``_get_retriever`` caches by experiment name across calls (real usage:
    one retriever per running server). Tests share the same mock experiment
    name, so without clearing this cache a retriever mocked in one test would
    leak into the next and mask the real assertion under test."""
    from retrieval import vqa

    vqa._RETRIEVER_CACHE.clear()
    yield
    vqa._RETRIEVER_CACHE.clear()


@pytest.fixture
def mock_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            frame_id=f"v1/f{i:04d}",
            video_id="v1",
            score=0.85 - i * 0.05,
            frame_path=str(Path("runs/test/frames/v1") / f"f{i:04d}.jpg"),
            video_name="test.mp4",
            shot_id="s1",
            frame_index=i,
            timestamp_sec=i * 2.0,
        )
        for i in range(5)
    ]


class TestVqaSearch:
    """Test the VQA pipeline orchestrator."""

    def test_vqa_search_returns_expected_structure(self, mock_experiment, mock_search_results):
        from retrieval.vqa import vqa_search
        from retrieval.temporal_search import ShotInput

        with (
            patch("retrieval.vqa._get_retriever") as mock_get_retriever,
            patch("retrieval.vqa.load_temporal_data") as mock_load,
            patch("retrieval.vqa.find_segments") as mock_find_seg,
            patch("retrieval.vqa.gather_frame_s") as mock_gather,
            patch("retrieval.vqa.create_agent") as mock_create_agent,
        ):
            mock_embeddings = np.random.rand(10, 512).astype("float32")
            mock_records = [
                {
                    "frame_id": "v1/f0000",
                    "video_id": "v1",
                    "frame_index": i,
                    "timestamp_sec": i * 2.0,
                    "frame_path": str(Path(f"frames/v1/f{i:04d}.jpg")),
                    "shot_id": "s1",
                }
                for i in range(10)
            ]
            mock_load.return_value = (mock_embeddings, mock_records)

            mock_embedder = MagicMock()
            mock_embedder.embed_text.return_value = [0.1] * 512

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = mock_search_results
            mock_retriever.embedders = {"siglip2-large": mock_embedder}
            mock_get_retriever.return_value = mock_retriever

            mock_find_seg.return_value = [
                {"start_pos": 2, "end_pos": 6, "length": 5, "center_idx": 4}
            ]

            mock_shot = ShotInput(
                video_id="v1",
                video_name="test.mp4",
                frames=mock_records[2:7],
                frame_paths=[str(Path(f"frames/v1/f{i:04d}.jpg")) for i in range(2, 7)],
                frame_count=5,
                start_timestamp=4.0,
                end_timestamp=12.0,
                sim_score=0.75,
            )
            mock_gather.return_value = mock_shot

            mock_agent = MagicMock()
            mock_agent.answer.return_value = "The person is wearing a blue shirt."
            mock_create_agent.return_value = mock_agent

            result = vqa_search(
                experiment=mock_experiment,
                query="person wearing blue",
                question="What color is the shirt?",
                context="outdoor scene",
                top_k=5,
            )

        assert "answer" in result
        assert result["answer"] == "The person is wearing a blue shirt."
        assert "results" in result
        assert len(result["results"]) == 5
        assert "pipeline" in result
        assert "embed_search" in result["pipeline"]
        assert "temporal_search" in result["pipeline"]
        assert "gather_shot" in result["pipeline"]
        assert "shot_validation" in result["pipeline"]
        assert "agent" in result["pipeline"]
        assert result["pipeline"]["agent"]["answer"] == "The person is wearing a blue shirt."
        assert "reasoning" in result

    def test_vqa_search_no_results(self, mock_experiment):
        from retrieval.vqa import vqa_search

        with patch("retrieval.vqa._get_retriever") as mock_get_retriever:
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []
            mock_get_retriever.return_value = mock_retriever

            result = vqa_search(
                experiment=mock_experiment,
                query="nothing",
                question="Any?",
                top_k=5,
            )

        assert result["answer"] == "No relevant frames found."
        assert result["results"] == []

    def test_vqa_search_temporal_fail_returns_fallback(self, mock_experiment, mock_search_results):
        from retrieval.vqa import vqa_search

        with (
            patch("retrieval.vqa._get_retriever") as mock_get_retriever,
            patch("retrieval.vqa.load_temporal_data", side_effect=FileNotFoundError("Missing")),
        ):
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = mock_search_results
            mock_get_retriever.return_value = mock_retriever

            result = vqa_search(
                experiment=mock_experiment,
                query="test",
                question="Q?",
                top_k=5,
            )

        assert "Could not find" in result["answer"] or "not" in result["answer"]
        assert len(result["results"]) == 5
