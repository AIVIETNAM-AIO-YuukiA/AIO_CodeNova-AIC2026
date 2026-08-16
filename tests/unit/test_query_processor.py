"""Tests for the query processor (Docker-LLM translate/expand + pass-through)."""

from unittest.mock import MagicMock

from retrieval.query_processor import (
    LlmQueryProcessor,
    PassThroughQueryProcessor,
    ProcessedQuery,
    _parse_bonus_weights,
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
    assert processed.weights == {"ocr_bonus": 0.0, "asr_bonus": 0.0, "caption_bonus": 0.0}


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
        ' "asr_keywords": [], "metadata": {},'
        ' "weights": {"caption_bonus": 0.2, "ocr_bonus": 0.0, "asr_bonus": 0.1}}'
    )
    processor._client = client

    processed = processor.process("xe hơi đỏ trên cao tốc")
    assert processed.weights == {"caption_bonus": 0.2, "ocr_bonus": 0.0, "asr_bonus": 0.1}


def test_llm_processor_ignores_legacy_weight_schema() -> None:
    # Older prompts answered with kis/ocr/asr shares that summed to 1.0. Those
    # keys mean nothing to the bonus model, so they must not leak through as
    # bonuses — every modality falls back to no bonus instead.
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = (
        '{"visual_prompt": "a red car on a highway", "ocr_keywords": [],'
        ' "asr_keywords": [], "metadata": {}, "weights": {"kis": 0.9, "ocr": 0.0, "asr": 0.1}}'
    )
    processor._client = client

    processed = processor.process("xe hơi đỏ trên cao tốc")
    assert processed.weights == {"caption_bonus": 0.0, "ocr_bonus": 0.0, "asr_bonus": 0.0}


def test_llm_processor_defaults_weights_when_missing() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = (
        '{"visual_prompt": "a cooking scene", "ocr_keywords": [], "asr_keywords": [], "metadata": {}}'
    )
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.weights == {"ocr_bonus": 0.0, "asr_bonus": 0.0, "caption_bonus": 0.0}


def test_parse_bonus_weights_keeps_in_range_values() -> None:
    result = _parse_bonus_weights(
        {"caption_bonus": 0.3, "ocr_bonus": 0.2, "asr_bonus": 0.1}
    )
    assert result == {"caption_bonus": 0.3, "ocr_bonus": 0.2, "asr_bonus": 0.1}


def test_parse_bonus_weights_clamps_to_half() -> None:
    # A runaway LLM weight must not be able to dominate the base visual score.
    result = _parse_bonus_weights({"caption_bonus": 5.0, "ocr_bonus": -1.0, "asr_bonus": 0.5})
    assert result == {"caption_bonus": 0.5, "ocr_bonus": 0.0, "asr_bonus": 0.5}


def test_parse_bonus_weights_defaults_missing_keys_to_zero() -> None:
    assert _parse_bonus_weights({}) == {
        "caption_bonus": 0.0,
        "ocr_bonus": 0.0,
        "asr_bonus": 0.0,
    }


def test_parse_bonus_weights_ignores_bad_types() -> None:
    result = _parse_bonus_weights(
        {"caption_bonus": "not-a-number", "ocr_bonus": 0.2, "asr_bonus": None}
    )
    assert result == {"caption_bonus": 0.0, "ocr_bonus": 0.2, "asr_bonus": 0.0}
