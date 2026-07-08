"""Tests for the Agent tab router and chat payload."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Experiment
from core.types import SearchResult


@pytest.fixture
def mock_experiment(tmp_path: Path) -> Experiment:
    run_dir = tmp_path / "runs" / "test"
    run_dir.mkdir(parents=True)
    config = MagicMock()
    config.clip_model = "ViT-B/32"
    config.embedding_model = "ViT-B/32"
    config.device = "cpu"
    config.data_dir = tmp_path / "data"
    return Experiment(name="test", run_dir=run_dir, config=config)


def test_agent_chat_requests_more_detail_for_short_query(mock_experiment):
    from ui.agent_tab import build_agent_payload

    retriever = MagicMock()
    sessions: dict[str, object] = {}

    result = build_agent_payload(
        experiment=mock_experiment,
        retriever=retriever,
        payload={"message": "help"},
        sessions=sessions,
        default_top_k=5,
    )

    assert result["needs_follow_up"] is True
    assert result["route"] == "ad_hoc"
    assert "detail" in result["follow_up"].lower() or "detail" in result["reply"].lower()
    retriever.search.assert_not_called()


def test_agent_chat_routes_vqa_and_returns_pipeline(mock_experiment):
    from ui.agent_tab import build_agent_payload

    retriever = MagicMock()
    sessions: dict[str, object] = {}

    with patch("ui.agent_tab.vqa_search") as mock_vqa:
        mock_vqa.return_value = {
            "answer": "The shirt is blue.",
            "results": [
                SearchResult(
                    frame_id="v1/f0001",
                    video_id="v1",
                    score=0.91,
                    frame_path=str(Path("runs/test/frames/v1/f0001.jpg")),
                    video_name="demo.mp4",
                    shot_id="s1",
                    frame_index=1,
                    timestamp_sec=2.0,
                ).to_dict(),
            ],
            "pipeline": {"agent": {"answer": "The shirt is blue."}},
        }

        result = build_agent_payload(
            experiment=mock_experiment,
            retriever=retriever,
            payload={"message": "What color is the shirt?"},
            sessions=sessions,
            default_top_k=5,
        )

    assert result["route"] == "vqa"
    assert result["reply"] == "The shirt is blue."
    assert result["results"]
    assert result["pipeline"]["agent"]["answer"] == "The shirt is blue."
    retriever.search.assert_not_called()


def test_agent_chat_routes_trake_and_flattens_evidence(mock_experiment):
    from ui.agent_tab import build_agent_payload

    retriever = MagicMock()
    sessions: dict[str, object] = {}

    with patch("ui.agent_tab.trake_search") as mock_trake:
        mock_trake.return_value = {
            "videos": [
                {
                    "video_id": "v1",
                    "video_name": "demo.mp4",
                    "score": 0.77,
                    "events": [
                        {
                            "frame_id": "v1/f0010",
                            "frame_path": str(Path("runs/test/frames/v1/f0010.jpg")),
                            "timestamp_sec": 4.0,
                            "frame_index": 10,
                            "shot_id": "s1",
                        },
                        {
                            "frame_id": "v1/f0020",
                            "frame_path": str(Path("runs/test/frames/v1/f0020.jpg")),
                            "timestamp_sec": 8.0,
                            "frame_index": 20,
                            "shot_id": "s2",
                        },
                    ],
                }
            ]
        }

        result = build_agent_payload(
            experiment=mock_experiment,
            retriever=retriever,
            payload={"message": "person enters room then leaves room"},
            sessions=sessions,
            default_top_k=5,
        )

    assert result["route"] == "trake"
    assert result["videos"]
    assert result["results"]
    assert result["results"][0]["image_url"].startswith("/frame?path=")
    retriever.search.assert_not_called()