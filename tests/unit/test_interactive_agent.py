"""Tests for the interactive search agent (AIC_2025-style narrowing loop)."""

import json
from unittest.mock import MagicMock

from agent.interactive import (
    MAX_TOOL_ROUNDS,
    InteractiveAgent,
    _parse_tool_json,
    _summarize_for_model,
)


def _agent_with_client(responses: list[str]) -> tuple[InteractiveAgent, MagicMock]:
    agent = InteractiveAgent(experiment=MagicMock(name="exp"))
    client = MagicMock()
    client.complete_text.side_effect = responses
    agent._client = client
    return agent, client


class TestTurnLoop:
    def test_plain_message_ends_turn(self) -> None:
        agent, _ = _agent_with_client(
            ['{"thought": "done", "message": "Bạn bấm vào ảnh để nộp nhé."}']
        )
        result = agent.run_turn([{"role": "user", "content": "tìm thấy rồi"}])
        assert result["done"] is True
        assert "nộp" in result["message"]

    def test_ask_user_short_circuits(self) -> None:
        agent, client = _agent_with_client(
            [
                json.dumps(
                    {
                        "thought": "cần hỏi",
                        "tool": "ask_user",
                        "args": {
                            "question": "Trong trường quay hay ngoài trời?",
                            "suggestions": ["Trong trường quay", "Ngoài trời"],
                        },
                    }
                )
            ]
        )
        result = agent.run_turn([{"role": "user", "content": "một người dẫn chương trình"}])
        assert result["done"] is False
        assert result["question"] == "Trong trường quay hay ngoài trời?"
        assert result["suggestions"] == ["Trong trường quay", "Ngoài trời"]
        assert client.complete_text.call_count == 1

    def test_search_then_ask(self) -> None:
        agent, client = _agent_with_client(
            [
                '{"tool": "search_kis", "args": {"query": "news anchor", "num_results": 5}}',
                '{"tool": "ask_user", "args": {"question": "Video nào giống nhất?"}}',
            ]
        )
        hit = MagicMock()
        hit.to_dict.return_value = {
            "frame_id": "f1",
            "video_id": "v1",
            "video_name": "L01_V001",
            "timestamp_sec": 3.0,
            "score": 0.9,
            "caption": None,
        }
        retriever = MagicMock()
        retriever.search.return_value = [hit]
        agent._retriever = retriever

        result = agent.run_turn([{"role": "user", "content": "người dẫn chương trình"}])
        assert result["done"] is False
        assert result["results"][0]["video_name"] == "L01_V001"
        assert [a["tool"] for a in result["actions"]] == ["search_kis", "ask_user"]
        # tool observation must have been fed back to the model
        convo = client.complete_text.call_args.kwargs["extra_messages"]
        assert any("KẾT QUẢ TOOL search_kis" in m["content"] for m in convo)

    def test_round_budget_exhausts(self) -> None:
        agent, client = _agent_with_client(
            ['{"tool": "search_kis", "args": {"query": "x"}}'] * MAX_TOOL_ROUNDS
        )
        retriever = MagicMock()
        retriever.search.return_value = []
        agent._retriever = retriever
        result = agent.run_turn([{"role": "user", "content": "q"}])
        assert result["done"] is True
        assert client.complete_text.call_count == MAX_TOOL_ROUNDS

    def test_llm_down_returns_hint(self) -> None:
        agent = InteractiveAgent(experiment=MagicMock(name="exp"))
        client = MagicMock()
        client.complete_text.side_effect = ConnectionError("refused")
        agent._client = client
        result = agent.run_turn([{"role": "user", "content": "q"}])
        assert result["done"] is True
        assert "OPENROUTER_MODEL" in result["message"]


class TestHelpers:
    def test_parse_tool_json_extracts_object(self) -> None:
        data = _parse_tool_json('bla {"tool": "search_kis", "args": {}} bla')
        assert data["tool"] == "search_kis"
        assert _parse_tool_json("no json here") is None

    def test_summarize_for_model_distribution(self) -> None:
        results = [
            {
                "video_name": "L01_V001",
                "timestamp_sec": 1.0,
                "score": 0.9,
                "caption": "phát thanh viên",
            },
            {"video_name": "L01_V001", "timestamp_sec": 5.0, "score": 0.8},
            {"video_name": "L02_V003", "timestamp_sec": 9.0, "score": 0.7, "text": "tin nóng"},
        ]
        summary = _summarize_for_model(results)
        assert "L01_V001 (2)" in summary
        assert "phát thanh viên" in summary
        assert "tin nóng" in summary
        assert _summarize_for_model([]) == "Không có kết quả nào."

    def test_subagent_without_results(self) -> None:
        agent = InteractiveAgent(experiment=MagicMock(name="exp"))
        message = agent._subagent_summarize("focus", [])
        assert "search" in message
