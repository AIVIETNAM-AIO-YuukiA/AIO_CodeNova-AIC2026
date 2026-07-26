"""Tests for the query processor (Docker-LLM translate/expand + pass-through)."""

from unittest.mock import MagicMock

from retrieval.query_processor import (
    LlmQueryProcessor,
    PassThroughQueryProcessor,
    ProcessedQuery,
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


def test_get_query_processor_returns_llm_processor() -> None:
    assert isinstance(get_query_processor(), LlmQueryProcessor)


def test_llm_processor_parses_json_response() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = (
        '{"visual_prompt": "a cooking scene", "ocr_keywords": ["bếp"],'
        ' "asr_keywords": ["nấu ăn"], "metadata": {"location_type": "indoor"}}'
    )
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.visual_prompt == "a cooking scene"
    assert processed.ocr_keywords == ["bếp"]
    assert processed.asr_keywords == ["nấu ăn"]
    assert processed.metadata == {"location_type": "indoor"}


def test_llm_processor_falls_back_and_disables_on_error() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.side_effect = ConnectionError("refused")
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.visual_prompt == "cảnh nấu ăn"
    assert processor._disabled is True

    # Second call must not hit the server again
    client.complete_text.reset_mock()
    processor.process("một query khác")
    client.complete_text.assert_not_called()


def test_llm_processor_falls_back_on_garbage_output() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = "not json at all"
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.visual_prompt == "cảnh nấu ăn"
