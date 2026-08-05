"""Tests for the Agent package (Docker-LLM brain + VLM tools + ReAct loop)."""

from unittest.mock import MagicMock

from agent.brain import AgentBrain, BrainResponse, parse_brain_response
from agent.react import Agent, create_agent
from agent.tools import CaptionTool, OCRTool, default_tools
from retrieval.temporal_search import ShotInput


class TestParseBrainResponse:
    def test_with_answer(self) -> None:
        text = '{"thought": "I see a cat", "answer": "A cat", "finished": true}'
        result = parse_brain_response(text)
        assert result.finished is True
        assert result.answer == "A cat"

    def test_with_action(self) -> None:
        text = '{"thought": "Need to look closer", "action": "caption", "action_input": {"image_path": "f.jpg"}}'
        result = parse_brain_response(text)
        assert result.finished is False
        assert result.action == "caption"
        assert result.action_input == {"image_path": "f.jpg"}

    def test_no_json_is_final_answer(self) -> None:
        result = parse_brain_response("I think the answer is 42")
        assert result.finished is True
        assert "42" in result.answer

    def test_invalid_json_is_final_answer(self) -> None:
        result = parse_brain_response("{not valid json")
        assert result.finished is True


class TestBrain:
    def test_reason_parses_client_response(self) -> None:
        brain = AgentBrain(model_name="test-model")
        client = MagicMock()
        client.complete_text.return_value = (
            '{"thought": "t", "answer": "Xã Diên Điền", "finished": true}'
        )
        brain._client = client
        response = brain.reason(question="Tên xã?", shot_info="video=v1", frame_count=3)
        assert response.finished is True
        assert response.answer == "Xã Diên Điền"
        # tool history must be forwarded to the LLM on later steps
        brain.reason(
            question="q",
            shot_info="s",
            frame_count=1,
            tool_results=[{"tool": "ocr", "result": "some text"}],
        )
        prompt = client.complete_text.call_args.kwargs["user_prompt"]
        assert "some text" in prompt

    def test_reason_server_down_finishes_with_hint(self) -> None:
        brain = AgentBrain(model_name="test-model")
        client = MagicMock()
        client.complete_text.side_effect = ConnectionError("refused")
        brain._client = client
        response = brain.reason(question="q", shot_info="s", frame_count=1)
        assert response.finished is True
        assert "AGENT_LOCAL_ENGINE_URL" in response.answer


class TestTools:
    def test_default_tools_are_caption_and_ocr_only(self) -> None:
        tools = default_tools()
        assert set(tools) == {"caption", "ocr"}

    def test_caption_tool_missing_file(self) -> None:
        result = CaptionTool().run(image_path="/nonexistent/frame.jpg")
        assert "not found" in result

    def test_ocr_tool_server_down_degrades(self, tmp_path) -> None:
        image = tmp_path / "f.jpg"
        image.write_bytes(b"fake")
        tool = OCRTool(base_url="http://localhost:1/v1")
        result = tool.run(image_path=str(image))
        assert "unavailable" in result.lower()


def _shot() -> ShotInput:
    return ShotInput(
        video_id="v1",
        video_name="L01_V001",
        frames=[{"frame_id": "f1"}],
        frame_paths=["/tmp/f1.jpg"],
        frame_count=1,
        start_timestamp=1.0,
        end_timestamp=2.0,
    )


class TestAgentLoop:
    def test_answer_returns_on_finished(self) -> None:
        brain = MagicMock()
        brain.reason.return_value = BrainResponse(answer="42", finished=True)
        agent = Agent(brain=brain, tools={})
        assert agent.answer(shot=_shot(), question="q") == "42"

    def test_answer_runs_tool_then_finishes(self) -> None:
        brain = MagicMock()
        brain.reason.side_effect = [
            BrainResponse(action="ocr", action_input={}, finished=False),
            BrainResponse(answer="done", finished=True),
        ]
        tool = MagicMock()
        tool.run.return_value = "ocr text"
        agent = Agent(brain=brain, tools={"ocr": tool})
        assert agent.answer(shot=_shot(), question="q") == "done"
        # image_path auto-injected from the shot's centre frame
        assert tool.run.call_args.kwargs["image_path"] == "/tmp/f1.jpg"

    def test_answer_context_reaches_brain(self) -> None:
        brain = MagicMock()
        brain.reason.return_value = BrainResponse(answer="ok", finished=True)
        agent = Agent(brain=brain, tools={})
        agent.answer(shot=_shot(), question="q", context="- cached caption")
        assert "cached caption" in brain.reason.call_args.kwargs["shot_info"]

    def test_max_steps_forces_fallback(self) -> None:
        brain = MagicMock()
        brain.reason.return_value = BrainResponse(action="nope", finished=False)
        agent = Agent(brain=brain, tools={}, max_steps=2)
        answer = agent.answer(shot=_shot(), question="q")
        assert "Question: q" in answer


def test_create_agent_always_uses_docker_brain() -> None:
    agent = create_agent(backend="gemini")
    assert isinstance(agent.brain, AgentBrain)
