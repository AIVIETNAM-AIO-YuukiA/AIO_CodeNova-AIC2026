"""Unit tests for InternVL agent integration."""

from unittest.mock import patch

import pytest
import torch

from agent.internvl import (
    InternVLBrain,
    InternVLCaptionTool,
    InternVLOCRTool,
    build_transform,
    dynamic_preprocess,
    internvl_default_tools,
)

# --- Mocks ---


class MockTokenizer:
    """Mock tokenizer."""

    pass


class MockModel:
    """Mock InternVL model."""

    device = torch.device("cpu")

    def __init__(self):
        self.device = torch.device("cpu")

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config=None,
        history=None,
        return_history=False,
    ):
        """Simulate model chat response based on input prompts."""
        if "Describe this image" in question:
            response = "A detailed description of the mock image."
        elif "Extract all visible text" in question:
            response = "MOCK TEXT EXTRACTED"
        elif "Please generate a valid JSON" in question:
            # ReAct response
            if "mock_action" in question:
                response = '{"thought": "I need to check", "action": "ocr", "action_input": {"image_path": "mock.jpg"}}'
            else:
                response = (
                    '{"thought": "I know the answer", "answer": "Mock answer", "finished": true}'
                )
        else:
            response = "Mock fallback response"

        if return_history:
            return response, []
        return response

    def eval(self):
        return self


@pytest.fixture
def mock_internvl_loader():
    """Mock the get_internvl_model_and_tokenizer function to avoid loading real 5GB model."""
    with patch("agent.internvl.get_internvl_model_and_tokenizer") as mock_loader:
        mock_loader.return_value = (MockModel(), MockTokenizer())
        yield mock_loader


@pytest.fixture
def mock_image_path(tmp_path):
    """Create a temporary dummy image for testing."""
    from PIL import Image

    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (800, 600), color="red")
    img.save(img_path)
    return str(img_path)


# --- Tests ---


class TestInternVLPreprocessing:
    def test_build_transform(self):
        transform = build_transform(input_size=448)
        assert transform is not None

    def test_dynamic_preprocess(self, tmp_path):
        from PIL import Image

        img = Image.new("RGB", (1920, 1080), color="blue")
        processed = dynamic_preprocess(
            img, min_num=1, max_num=6, image_size=448, use_thumbnail=True
        )

        # Should return a list of cropped blocks + 1 thumbnail
        assert isinstance(processed, list)
        assert len(processed) > 1
        assert processed[0].size == (448, 448)


class TestInternVLBrain:
    def test_reason_answer(self, mock_internvl_loader):
        brain = InternVLBrain()
        response = brain.reason(
            question="What is this?", shot_info="Mock shot", frame_count=1, tool_results=[]
        )
        assert response.finished is True
        assert response.answer == "Mock answer"

    def test_reason_action(self, mock_internvl_loader):
        brain = InternVLBrain()
        # The mock model returns an action JSON when "mock_action" is in the prompt
        response = brain.reason(
            question="mock_action", shot_info="Mock shot", frame_count=1, tool_results=[]
        )
        assert response.finished is False
        assert response.action == "ocr"
        assert response.action_input == {"image_path": "mock.jpg"}

    def test_parse_response_no_json(self):
        brain = InternVLBrain()
        response = brain._parse_response("This is not JSON.")
        assert response.finished is True
        assert response.answer == "This is not JSON."


class TestInternVLTools:
    def test_caption_tool_run(self, mock_internvl_loader, mock_image_path):
        tool = InternVLCaptionTool()
        result = tool.run(image_path=mock_image_path)
        assert result == "A detailed description of the mock image."

    def test_caption_tool_missing_file(self):
        tool = InternVLCaptionTool()
        result = tool.run(image_path="nonexistent.jpg")
        assert "Error: file not found" in result

    def test_caption_tool_no_path(self):
        tool = InternVLCaptionTool()
        result = tool.run(image_path="")
        assert "Error: image_path is required" in result

    def test_ocr_tool_run(self, mock_internvl_loader, mock_image_path):
        tool = InternVLOCRTool()
        result = tool.run(image_path=mock_image_path)
        assert result == "MOCK TEXT EXTRACTED"

    def test_ocr_tool_missing_file(self):
        tool = InternVLOCRTool()
        result = tool.run(image_path="nonexistent.jpg")
        assert "Error: file not found" in result


def test_internvl_default_tools():
    tools = internvl_default_tools()
    assert "caption" in tools
    assert "ocr" in tools
    assert "detect" in tools
    assert "asr" in tools
    assert isinstance(tools["caption"], InternVLCaptionTool)
    assert isinstance(tools["ocr"], InternVLOCRTool)
