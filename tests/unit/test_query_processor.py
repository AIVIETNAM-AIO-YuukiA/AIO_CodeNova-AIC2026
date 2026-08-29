"""Tests for the query processor (OpenRouter LLM translate/expand + pass-through)."""

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
    assert processed.weights == {"ocr_bonus": 0.0, "asr_bonus": 0.0}
    assert processed.routing_mode == "heuristic"
    assert processed.modality_confidence == {
        "kis": 1.0,
        "ocr": 0.0,
        "asr": 0.0,
    }


def test_get_query_processor_returns_llm_processor() -> None:
    assert isinstance(get_query_processor(), LlmQueryProcessor)


def test_llm_processor_parses_json_response() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = (
        '{"components": {"kis": "a cooking scene", "ocr": "bếp", "asr": "nấu ăn"},'
        ' "weights": {"kis": 0.6, "ocr": 0.2, "asr": 0.2}, "reasoning": "test"}'
    )
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.visual_prompt == "a cooking scene"
    assert processed.ocr_keywords == ["bếp"]
    assert processed.asr_keywords == ["nấu ăn"]
    assert processed.routing_mode == "llm"
    assert processed.llm_status == "ok"
    assert processed.llm_calls == 1
    assert processed.llm_attempts == 1
    assert processed.modality_confidence["ocr"] == 0.2
    assert processed.modality_confidence["asr"] == 0.2


def test_llm_processor_falls_back_per_request_then_opens_circuit() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.side_effect = ConnectionError("refused")
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.visual_prompt == "cảnh nấu ăn"
    assert processed.routing_mode == "fallback"
    assert processed.llm_status == "error"
    assert processed.llm_calls == 2
    assert processed.llm_attempts == 2
    assert processor._disabled is False

    # A single failed request no longer disables the LLM for the whole session.
    processor.process("một query khác")
    assert client.complete_text.call_count == 4

    # Three failed requests open the circuit. The next request falls back
    # without touching the backend.
    processor.process("query thứ ba")
    assert processor._disabled is True
    client.complete_text.reset_mock()
    blocked = processor.process("query thứ tư")
    client.complete_text.assert_not_called()
    assert blocked.llm_status == "circuit_open"
    assert blocked.llm_calls == 0


def test_llm_processor_falls_back_on_garbage_output() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = "not json at all"
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.visual_prompt == "cảnh nấu ăn"
    assert processed.routing_mode == "fallback"
    assert processed.llm_status == "error"
    assert processed.llm_calls == 1


def test_llm_processor_derives_weights_from_confidence() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = (
        '{"components": {"kis": "a red car on a highway", "ocr": "EXIT", "asr": "traffic"},'
        ' "weights": {"kis": 0.4, "ocr": 0.4, "asr": 0.2}, "reasoning": "test"}'
    )
    processor._client = client

    processed = processor.process("xe hơi đỏ trên cao tốc")
    assert processed.modality_confidence == {
        "kis": 0.5,
        "ocr": 0.4,
        "asr": 0.2,
    }
    assert processed.weights == {"ocr_bonus": 0.2, "asr_bonus": 0.1}


def test_llm_processor_defaults_weights_when_missing() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = (
        '{"components": {"kis": "a cooking scene", "ocr": null, "asr": null}}'
    )
    processor._client = client

    processed = processor.process("cảnh nấu ăn")
    assert processed.weights == {"ocr_bonus": 0.0, "asr_bonus": 0.0}


def test_use_llm_false_routes_with_heuristics_without_calling_backend() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    processor._client = client

    processed = processor.process(
        'Tìm cảnh có biển hiệu ghi "GIẢM GIÁ", có người nói về khuyến mãi',
        use_llm=False,
    )

    client.complete_text.assert_not_called()
    assert processed.routing_mode == "heuristic"
    assert processed.llm_status == "disabled"
    assert processed.modality_confidence["kis"] == 1.0
    assert processed.modality_confidence["ocr"] >= 0.85
    assert processed.modality_confidence["asr"] >= 0.85
    assert processed.ocr_keywords == ["GIẢM GIÁ"]
    assert processed.asr_keywords == ["khuyến mãi"]
    assert processed.normalized_keywords["ocr"] == ["giảm giá", "giam gia"]
    assert processed.normalized_keywords["asr"] == ["khuyến mãi", "khuyen mai"]
    assert processed.weights["ocr_bonus"] > 0.0
    assert processed.weights["asr_bonus"] > 0.0


def test_heuristic_signal_matching_uses_word_boundaries() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    processed = processor.process("chung cư nổi bật", use_llm=False)

    # "chu" must not match the middle of "chung" and "noi" must not match
    # Vietnamese "nổi" after diacritic folding.
    assert processed.ocr_keywords == []
    assert processed.asr_keywords == []


def test_heuristic_separates_ocr_and_asr_clauses() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    processed = processor.process(
        "tìm cảnh có biển hiệu hoặc chữ quảng cáo trên màn hình, có người nói về giảm giá",
        use_llm=False,
    )

    assert processed.ocr_keywords == ["quảng cáo"]
    assert processed.asr_keywords == ["giảm giá"]


def test_transient_failure_retries_once_then_succeeds() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.side_effect = [
        TimeoutError("slow"),
        '{"components": {"kis": "a market", "ocr": null, "asr": null}}',
    ]
    processor._client = client

    processed = processor.process("khu chợ")

    assert client.complete_text.call_count == 2
    assert processed.llm_status == "ok"
    assert processed.llm_calls == 2
    assert processor._consecutive_failures == 0


def test_non_transient_parse_error_is_not_retried() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.return_value = "not json"
    processor._client = client

    processed = processor.process("khu chợ")

    assert client.complete_text.call_count == 1
    assert processed.llm_status == "error"
    assert processed.llm_calls == 1


def test_circuit_breaker_half_open_after_cooldown() -> None:
    now = [100.0]
    processor = LlmQueryProcessor(
        model_name="test-model",
        circuit_failure_threshold=3,
        circuit_cooldown_seconds=60.0,
        monotonic=lambda: now[0],
    )
    client = MagicMock()
    client.complete_text.side_effect = ValueError("bad request")
    processor._client = client

    for query in ("q1", "q2", "q3"):
        assert processor.process(query).llm_status == "error"
    assert processor.process("blocked").llm_status == "circuit_open"

    now[0] += 60.0
    client.complete_text.side_effect = None
    client.complete_text.return_value = (
        '{"components": {"kis": "recovered", "ocr": null, "asr": null}}'
    )
    recovered = processor.process("probe")

    assert recovered.llm_status == "ok"
    assert recovered.visual_prompt == "recovered"
    assert processor._disabled is False
    assert processor._consecutive_failures == 0


def test_success_resets_failures_before_threshold() -> None:
    processor = LlmQueryProcessor(model_name="test-model")
    client = MagicMock()
    client.complete_text.side_effect = [
        ValueError("bad response"),
        '{"components": {"kis": "ok", "ocr": null, "asr": null}}',
    ]
    processor._client = client

    assert processor.process("first").llm_status == "error"
    assert processor._consecutive_failures == 1
    assert processor.process("second").llm_status == "ok"
    assert processor._consecutive_failures == 0
