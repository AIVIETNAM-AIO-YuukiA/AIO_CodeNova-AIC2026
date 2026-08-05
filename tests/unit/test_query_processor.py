"""Tests for the query processor (Docker-LLM translate/expand + pass-through)."""

from unittest.mock import MagicMock

from retrieval.query_processor import (
    LlmQueryProcessor,
    PassThroughQueryProcessor,
    ProcessedQuery,
    _normalize_weights,
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
    assert processed.weights == {"kis": 1.0, "ocr": 0.0, "asr": 0.0}


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


def test_llm_processor_parses_weights() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = (
        '{"visual_prompt": "a red car on a highway", "ocr_keywords": [],'
        ' "asr_keywords": [], "metadata": {}, "weights": {"kis": 0.9, "ocr": 0.0, "asr": 0.1}}'
    )
    processor._client = client

    processed = processor.process("xe hơi đỏ trên cao tốc")
    assert processed.weights == {"kis": 0.9, "ocr": 0.0, "asr": 0.1}


def test_llm_processor_defaults_weights_when_missing() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = (
        '{"visual_prompt": "a cooking scene", "ocr_keywords": [], "asr_keywords": [], "metadata": {}}'
    )
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.weights == {"kis": 1.0, "ocr": 0.0, "asr": 0.0}


def test_normalize_weights_sums_to_one() -> None:
    result = _normalize_weights({"kis": 0.7, "ocr": 0.2, "asr": 0.1})
    assert result == {"kis": 0.7, "ocr": 0.2, "asr": 0.1}


def test_normalize_weights_rescales_when_not_summing_to_one() -> None:
    result = _normalize_weights({"kis": 1.4, "ocr": 0.4, "asr": 0.2})
    assert abs(sum(result.values()) - 1.0) < 1e-9
    assert result["kis"] == 0.7


def test_normalize_weights_falls_back_when_all_zero() -> None:
    assert _normalize_weights({"kis": 0.0, "ocr": 0.0, "asr": 0.0}) == {
        "kis": 1.0, "ocr": 0.0, "asr": 0.0,
    }


def test_normalize_weights_ignores_bad_types() -> None:
    result = _normalize_weights({"kis": "not-a-number", "ocr": 0.5, "asr": 0.5})
    assert result == {"kis": 0.0, "ocr": 0.5, "asr": 0.5}
