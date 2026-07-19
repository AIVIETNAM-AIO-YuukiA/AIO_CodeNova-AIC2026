import sys
from unittest.mock import MagicMock

# Mock google package and submodules so that the tests can run without installing google-genai
from types import ModuleType
google_mock = ModuleType("google")
genai_mock = ModuleType("google.genai")
genai_types_mock = ModuleType("google.genai.types")

mock_client_instance = MagicMock()
genai_mock.Client = MagicMock(return_value=mock_client_instance)

sys.modules["google"] = google_mock
sys.modules["google.genai"] = genai_mock
sys.modules["google.genai.types"] = genai_types_mock
google_mock.genai = genai_mock
genai_mock.types = genai_types_mock
genai_types_mock.GenerateContentConfig = MagicMock()

from retrieval.query_processor import (
    ProcessedQuery,
    PassThroughQueryProcessor,
    LlmQueryProcessor,
    get_query_processor,
)


def test_pass_through_query_processor() -> None:
    processor = PassThroughQueryProcessor()
    processed = processor.process("cảnh nấu ăn")
    
    assert isinstance(processed, ProcessedQuery)
    assert processed.raw_query == "cảnh nấu ăn"
    assert processed.visual_prompt == "cảnh nấu ăn"
    assert processed.ocr_keywords == []
    assert processed.asr_keywords == []
    assert processed.metadata == {}


def test_get_query_processor_defaults_to_passthrough(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    processor = get_query_processor()
    assert isinstance(processor, PassThroughQueryProcessor)


def test_get_query_processor_resolves_to_llm_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    processor = get_query_processor()
    assert isinstance(processor, LlmQueryProcessor)
    assert processor.api_key == "test_key"


def test_llm_query_processor_fallback_on_init_failure() -> None:
    # Reset mock to simulate import/config failure
    genai_mock.Client.side_effect = Exception("Config error")
    
    processor = LlmQueryProcessor(api_key="test_key")
    processed = processor.process("cảnh nấu ăn")
    
    assert processed.raw_query == "cảnh nấu ăn"
    assert processed.visual_prompt == "cảnh nấu ăn"
    
    # Restore mock
    genai_mock.Client.side_effect = None


def test_llm_query_processor_success() -> None:
    mock_response = MagicMock()
    mock_response.text = """
    {
        "visual_prompt": "A person preparing food in the kitchen",
        "ocr_keywords": ["kitchen", "food"],
        "asr_keywords": ["hello"],
        "metadata": {"action": "cooking"}
    }
    """
    
    mock_client_instance.models.generate_content.return_value = mock_response
    genai_mock.Client.reset_mock()
    
    processor = LlmQueryProcessor(api_key="test_key")
    processed = processor.process("cảnh nấu ăn")
    
    genai_mock.Client.assert_called_once_with(api_key="test_key")
    assert processed.raw_query == "cảnh nấu ăn"
    assert processed.visual_prompt == "A person preparing food in the kitchen"
    assert processed.ocr_keywords == ["kitchen", "food"]
    assert processed.asr_keywords == ["hello"]
    assert processed.metadata == {"action": "cooking"}


def test_llm_query_processor_json_failure_fallback() -> None:
    # If the response from the LLM is not valid JSON, it should fallback to returning raw query
    mock_response = MagicMock()
    mock_response.text = "invalid json response"
    
    mock_client_instance.models.generate_content.return_value = mock_response

    processor = LlmQueryProcessor(api_key="test_key")
    processed = processor.process("cảnh nấu ăn")
    
    assert processed.raw_query == "cảnh nấu ăn"
    assert processed.visual_prompt == "cảnh nấu ăn"
