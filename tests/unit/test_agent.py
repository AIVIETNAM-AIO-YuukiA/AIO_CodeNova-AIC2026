"""Tests for VLM Agent package."""

from unittest.mock import patch


from agent.brain import VlmBrain, BrainResponse
from agent.tools import CaptionTool, OCRTool, DetectTool, ASRTool, default_tools
from agent.react import Agent


class TestBrain:
    def test_parse_response_with_answer(self) -> None:
        brain = VlmBrain()
        text = '{"thought": "I see a cat", "answer": "A cat", "finished": true}'
        result = brain._parse_response(text)
        assert result.finished is True
        assert result.answer == "A cat"

    def test_parse_response_with_action(self) -> None:
        brain = VlmBrain()
        text = '{"thought": "Need to look closer", "action": "caption", "action_input": {"image_path": "f.jpg"}}'
        result = brain._parse_response(text)
        assert result.finished is False
        assert result.action == "caption"
        assert result.action_input == {"image_path": "f.jpg"}

    def test_parse_response_no_json(self) -> None:
        brain = VlmBrain()
        result = brain._parse_response("I think the answer is 42")
        assert result.finished is True
        assert "42" in result.answer

    def test_reason_no_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        brain = VlmBrain()
        response = brain.reason(question="test", shot_info="test", frame_count=1)
        assert response.finished is True


class TestTools:
    def test_caption_tool_no_key(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        tool = CaptionTool()
        result = tool.run(image_path="test.jpg")
        assert "unavailable" in result.lower() or "error" in result.lower()

    def test_ocr_tool_no_key(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        tool = OCRTool()
        result = tool.run(image_path="test.jpg")
        assert "unavailable" in result.lower()

    def test_detect_tool_no_ultralytics(self) -> None:
        tool = DetectTool()
        result = tool.run(image_path="test.jpg")
        assert "unavailable" in result.lower()

    def test_asr_tool_no_whisper(self) -> None:
        tool = ASRTool()
        result = tool.run(audio_path="test.mp3")
        assert "unavailable" in result.lower()

    def test_default_tools(self) -> None:
        tools = default_tools()
        assert "caption" in tools
        assert "ocr" in tools
        assert "detect" in tools
        assert "asr" in tools

    def test_caption_tool_no_path(self) -> None:
        tool = CaptionTool()
        result = tool.run()
        assert "required" in result


class TestAgent:
    def test_create_agent(self) -> None:
        agent = Agent()
        assert agent.max_steps == 5
        assert "caption" in agent.tools

    @patch("agent.brain.VlmBrain.reason")
    def test_agent_returns_answer(self, mock_reason) -> None:
        mock_reason.return_value = BrainResponse(
            thought="I can see a cat",
            answer="A cat is sitting on the table",
            finished=True,
        )

        from retrieval.temporal_search import ShotInput

        shot = ShotInput(
            video_id="v1", video_name="test.mp4", frame_paths=["f1.jpg"], frame_count=1
        )

        agent = Agent()
        answer = agent.answer(shot=shot, question="What is in the image?")

        assert "cat" in answer.lower()

    @patch("agent.brain.VlmBrain.reason")
    def test_agent_react_loop(self, mock_reason) -> None:
        """Test ReAct loop: brain asks for caption, then answers."""
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return BrainResponse(
                    thought="Need to examine the image",
                    action="caption",
                    action_input={"image_path": "f1.jpg"},
                    finished=False,
                )
            return BrainResponse(
                thought="Now I can see it",
                answer="A red car",
                finished=True,
            )

        mock_reason.side_effect = side_effect

        from retrieval.temporal_search import ShotInput

        shot = ShotInput(
            video_id="v1", video_name="test.mp4", frame_paths=["f1.jpg"], frame_count=1
        )

        agent = Agent(max_steps=3)
        answer = agent.answer(shot=shot, question="What color is the car?")

        assert "red" in answer.lower() or "car" in answer.lower()
        assert call_count[0] == 2
