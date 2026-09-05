"""Query processor components for translation, slot filling, and expansion."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)


class _LlmCallError(RuntimeError):
    """Internal wrapper that preserves how many backend calls were attempted."""

    def __init__(self, cause: Exception, calls: int) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.calls = calls


# Ported verbatim from the AIC_2025 reference project's
# ``GPT4o_service.analyze_intelligent_query`` system prompt.
_SYSTEM_PROMPT = """You are an expert Query Strategist for a multimedia retrieval system.
    Your task is to parse a user's query into a precise, weighted, multimodal search plan.
    You must differentiate between:
    1.  *Visual Concepts* (What you SEE: actions, objects, scenes) → KIS
    2.  *On-Screen Text* (What you READ: banners, names, numbers) → OCR
    3.  *Spoken Keywords* (What you HEAR: topics, lyrics, speech) → ASR

    --- COMPONENT & LANGUAGE RULES ---

    1.  *KIS (Visual Concepts):*
        * Describes only the visual elements: actions, scenes, objects, colors, settings.
        * ALWAYS translate to English for the SigLIP embedding model.
        * Example: "cô gái hát" → "woman singing", "người phát biểu" → "man speaking at podium".

    2.  *OCR (On-Screen Text):*
        * Contains only the exact text, numbers, or names likely visible on-screen (e.g., banners, logos, subtitles, scores, jersey names).
        * Keep ORIGINAL language for Elasticsearch exact matching.
        * Example: "Ronaldo" → "Cristiano Ronaldo 7", "đồng hồ 10 phút" → "10:00 10 phút".

    3.  *ASR (Spoken Keywords):*
        * Contains only the specific keywords, topics, or lyrics likely spoken in the audio.
        * NEVER include verbs describing the speech/singing (like "phát biểu", "hát bài").
        * Keep ORIGINAL language for Elasticsearch exact matching.
        * Example: "phát biểu về AI" → "AI trí tuệ nhân tạo", "hát bài Despacito" → "Despacito".

    --- STRATEGIC WEIGHTING HEURISTICS (Total must sum to 1.0) ---

    1.  *HEURISTIC A: VISUAL-DRIVEN (KIS High)*
        * Query is dominated by actions, attributes, or scenes.
        * Example: "người mặc áo đỏ chạy bộ trên bãi biển" → KIS: 0.9, OCR: 0.0, ASR: 0.1 (background noise).

    2.  *HEURISTIC B: TEXT-DRIVEN (OCR High)*
        * Query contains specific text, numbers, or scores that are precise signals.
        * Example: "video có đồng hồ hiện 10:30" → OCR: 0.7, KIS: 0.3 (visual context), ASR: 0.0.

    3.  *HEURISTIC C: AUDIO-DRIVEN (ASR High)*
        * Query is dominated by spoken topics, dialogue, or song lyrics.
        * Example: "bài phát biểu về kinh tế vĩ mô" → ASR: 0.7, KIS: 0.2 (visual is just 'man talking'), OCR: 0.1.

    4.  *HEURISTIC D: HYBRID (Balanced)*
        * This is the most common case. Distribute weight based on the query's components.
        * Example: "Thủ tướng Phạm Minh Chính (OCR/ASR) phát biểu về AI (ASR)"
            * Intent: Find a specific person (OCR) talking about a specific topic (ASR). Visual (KIS) is generic.
            * Weights: ASR: 0.5, OCR: 0.4, KIS: 0.1.
        * Example: "cô gái mặc áo đỏ (KIS) hát bài Despacito (ASR)"
            * Intent: Find a specific visual (KIS) combined with a specific lyric (ASR). OCR is irrelevant.
            * Weights: KIS: 0.5, ASR: 0.5, OCR: 0.0.
        * Example: "Cristiano Ronaldo (OCR) ghi bàn (KIS)"
            * Intent: Find a specific action (KIS) by a specific person (OCR, name on jersey). ASR (commentary) is secondary.
            * Weights: KIS: 0.6, OCR: 0.3, ASR: 0.1.

    --- EXAMPLES (Applying the new logic) ---

    Query: "Thủ tướng Phạm Minh Chính phát biểu về AI"
    - Components:
        "kis": "man speaking at podium, formal event, politician"
        "ocr": "Thủ tướng Phạm Minh Chính"
        "asr": "Thủ tướng Phạm Minh Chính trí tuệ nhân tạo AI"
    - Weights: {"kis": 0.1, "ocr": 0.4, "asr": 0.5}
    - Reasoning: "Primary intent is a spoken topic (ASR) by a specific entity (OCR). Visual (KIS) is generic."

    Query: "cô gái mặc áo đỏ hát bài Despacito"
    - Components:
        "kis": "woman wearing red dress singing, performance"
        "ocr": null
        "asr": "Despacito"
    - Weights: {"kis": 0.5, "ocr": 0.0, "asr": 0.5}
    - Reasoning: "Intent is balanced between a specific visual (KIS) and a specific song lyric (ASR)."

    Query: "Tìm cảnh nấu ăn có đồng hồ hiện 10 phút"
    - Components:
        "kis": "cooking scene in kitchen, preparing food"
        "ocr": "10:00 10 phút"
        "asr": null
    - Weights: {"kis": 0.6, "ocr": 0.4, "asr": 0.0}
    - Reasoning: "Intent is a visual scene (KIS) combined with a highly specific on-screen text (OCR)."
"""


@dataclass
class ProcessedQuery:
    """Structured representation of a processed search query."""

    raw_query: str
    visual_prompt: str  # English visual prompt
    visual_prompt_vi: str = ""  # Vietnamese visual prompt
    ocr_keywords: list[str] = field(default_factory=list)
    asr_keywords: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    # Bonus multipliers derived from routing confidence for backward-compatible
    # retrieval callers. Visual KIS is implicitly the base (1.0).
    weights: dict[str, float] = field(default_factory=lambda: {"ocr_bonus": 0.0, "asr_bonus": 0.0})
    # Additive routing/observability fields.  The original fields above remain
    # unchanged so older retrieval callers can continue to consume this type.
    routing_mode: str = "heuristic"  # llm | heuristic | fallback
    modality_confidence: dict[str, float] = field(
        default_factory=lambda: {"kis": 1.0, "ocr": 0.0, "asr": 0.0}
    )
    llm_status: str = "not_requested"  # ok | disabled | error | circuit_open
    fallback_reason: str | None = None
    normalized_keywords: dict[str, list[str]] = field(
        default_factory=lambda: {"ocr": [], "asr": []}
    )
    llm_calls: int = 0
    llm_attempts: int = 0
    llm_usage: dict[str, object] = field(default_factory=dict)


class QueryProcessor:
    """Base interface for query processing."""

    def process(
        self, query: str, enabled_models: list[str] | None = None, use_llm: bool = True
    ) -> ProcessedQuery:
        """Process a raw user query into structured fields."""
        raise NotImplementedError

    def extract_temporal_events(self, query: str, max_events: int = 5) -> list[str]:
        """Split a query into ordered events; empty list means "no sequence"."""
        return []

    def expand_query(self, visual_prompt: str, num_expansions: int = 4) -> list[str]:
        """Return extra visually-descriptive English query variants; empty if unavailable."""
        return []


class PassThroughQueryProcessor(QueryProcessor):
    """Query processor that passes the raw query directly without modifications."""

    def process(
        self, query: str, enabled_models: list[str] | None = None, use_llm: bool = True
    ) -> ProcessedQuery:
        return ProcessedQuery(raw_query=query, visual_prompt=query, visual_prompt_vi=query)


class LlmQueryProcessor(QueryProcessor):
    """Translate/expand queries via OpenRouter (OPENROUTER_MODEL_FOR_CHAT).

    The LLM augments deterministic routing rather than being a single point of
    failure.  A failed request falls back to Vietnamese/English heuristics for
    that request.  A small circuit breaker prevents repeated timeouts while
    still probing the backend again after a cooldown.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 60.0,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._model_name = model_name
        self._client = None
        self._failure_threshold = max(1, circuit_failure_threshold)
        self._cooldown_seconds = max(0.0, circuit_cooldown_seconds)
        self._monotonic = monotonic or time.monotonic
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None
        self._half_open_probe = False
        self._circuit_lock = threading.Lock()

    @property
    def _disabled(self) -> bool:
        """Compatibility view of the old flag: true only while the circuit is open."""
        with self._circuit_lock:
            if self._circuit_opened_at is None:
                return False
            return self._monotonic() - self._circuit_opened_at < self._cooldown_seconds

    def _load_client(self):
        if self._client is None:
            from modules._vllm_chat import VllmChatClient

            self._client = VllmChatClient(
                # Model text-only riêng cho việc mở rộng/phân tách câu truy vấn —
                # tách khỏi OPENROUTER_MODEL (dùng cho OCR/captioning, cần model
                # nhìn được ảnh) để đổi model chat không làm hỏng OCR.
                openrouter_model=self._model_name or os.environ.get("OPENROUTER_MODEL_FOR_CHAT"),
                openrouter_provider=os.environ.get("OPENROUTER_PROVIDER_FOR_CHAT"),
                # Query routing owns the single retry/circuit policy. Disable
                # the shared client's longer caption/OCR retry loop here.
                max_retries=0,
            )
        return self._client

    def process(
        self, query: str, enabled_models: list[str] | None = None, use_llm: bool = True
    ) -> ProcessedQuery:
        heuristic = _heuristic_processed_query(query)
        if use_llm is False:
            heuristic.llm_status = "disabled"
            return heuristic

        can_call, half_open = self._acquire_circuit()
        if not can_call:
            return _with_fallback(
                heuristic,
                llm_status="circuit_open",
                reason="LLM circuit breaker is open; using heuristic routing.",
            )

        is_vi = False
        if not enabled_models:
            is_vi = True
        else:
            for m in enabled_models:
                m_lower = m.lower()
                if "vietnamese" in m_lower or "vism" in m_lower:
                    is_vi = True
                else:
                    pass

        # Output shape ported from AIC_2025's analyze_intelligent_query
        # ("components" + "weights" + "reasoning"), extended with a
        # "components.kis_vi" field CodeNova needs for the Vietnamese
        # embedding branch — AIC_2025 has no Vietnamese embedder to feed.
        prompt = f"""Analyze this query: "{query}"
        {'Also provide "kis_vi": the same visual concept as "kis" but in Vietnamese (static noun-phrase, for a Vietnamese embedding model).' if is_vi else ""}
        Return JSON now:
        {{
            "components": {{
                "kis": "string",
                {'"kis_vi": "string",' if is_vi else ""}
                "ocr": "string or null",
                "asr": "string or null"
            }},
            "weights": {{"kis": 0.x, "ocr": 0.x, "asr": 0.x}},
            "reasoning": "string"
        }}
        """
        calls = 0
        try:
            client = self._load_client()
            if isinstance(getattr(client, "last_usage", None), dict):
                client.last_usage = {}
            text, calls = self._complete_with_retry(
                lambda: client.complete_text(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    generation_params={"temperature": 0.0, "max_tokens": 500},
                )
            )
            data = _extract_json(text)
            LOGGER.info("Qwen Preprocessing Result: %s", json.dumps(data, ensure_ascii=False))
            processed = _merge_llm_and_heuristic(
                query=query,
                data=data,
                heuristic=heuristic,
                llm_calls=calls,
                llm_usage=(
                    self._client.last_usage
                    if isinstance(getattr(self._client, "last_usage", None), dict)
                    else {}
                ),
            )
            self._record_success()
            return processed
        except Exception as exc:  # noqa: BLE001 - all LLM failures must fail open
            reported_exc = exc
            if isinstance(exc, _LlmCallError):
                calls = exc.calls
                reported_exc = exc.cause
            self._record_failure(half_open=half_open)
            LOGGER.warning(
                "LLM query processing failed: %s. Falling back to heuristic routing "
                "for this request.",
                reported_exc,
            )
            fallback = _with_fallback(
                heuristic,
                llm_status="error",
                reason=f"{type(reported_exc).__name__}: {reported_exc}"[:300],
            )
            fallback.llm_calls = calls
            fallback.llm_attempts = calls
            fallback.llm_usage = (
                dict(self._client.last_usage)
                if isinstance(getattr(self._client, "last_usage", None), dict)
                else {}
            )
            return fallback

    def _acquire_circuit(self) -> tuple[bool, bool]:
        """Return ``(allowed, half_open)`` and reserve the half-open probe if needed."""
        with self._circuit_lock:
            if self._circuit_opened_at is None:
                return True, False
            elapsed = self._monotonic() - self._circuit_opened_at
            if elapsed < self._cooldown_seconds or self._half_open_probe:
                return False, False
            self._half_open_probe = True
            return True, True

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures = 0
            self._circuit_opened_at = None
            self._half_open_probe = False

    def _record_failure(self, *, half_open: bool) -> None:
        with self._circuit_lock:
            self._half_open_probe = False
            if half_open:
                self._consecutive_failures = self._failure_threshold
            else:
                self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._circuit_opened_at = self._monotonic()

    def _complete_with_retry(self, call: Callable[[], str]) -> tuple[str, int]:
        """Run an LLM call, retrying at most once for a transient transport/upstream error."""
        calls = 0
        while True:
            calls += 1
            try:
                return call(), calls
            except Exception as exc:
                if calls >= 2 or not _is_transient_error(exc):
                    raise _LlmCallError(exc, calls) from exc
                LOGGER.info("Transient LLM query failure; retrying once: %s", exc)

    def extract_temporal_events(self, query: str, max_events: int = 5) -> list[str]:
        """Split one free-form query into an ordered list of atomic events.

        Lets a user type "a man walks out, then rides a motorbike, then turns
        left" instead of filling in the TRAKE event boxes by hand. Returns an
        empty list when the LLM is unavailable or the query holds no sequence,
        so callers can fall back to treating the query as a single event.

        Ported from the AIC_2025 reference project's
        ``GPT4o_service.extract_temporal_events``.
        """
        if not query.strip():
            return []

        can_call, half_open = self._acquire_circuit()
        if not can_call:
            return []

        system_prompt = (
            "You are a temporal event extractor for video retrieval.\n"
            f"Split the user's query into at most {max_events} atomic events in "
            "strict chronological order.\n"
            "Do NOT invent details or events not stated or strongly implied by the query.\n"
            "Write each event in concise English (<=12 words), present tense.\n"
            "Merge co-occurring visual attributes into the SAME event (colors, object "
            "details, prepositional phrases like 'with/on/in/at', Vietnamese 'với', "
            "'có', 'ở'). Do NOT split a scene and its attributes into separate events.\n"
            "If the query has N temporal connectors ('đầu tiên/tiếp theo/sau đó/cuối cùng', "
            "'first/then/next/finally'), usually return N+1 events.\n"
            'Return strict JSON: {"events": ["event 1", "event 2", ...]} with no extra text.'
        )
        user_prompt = f'User query:\n"""{query}"""\nReturn JSON now:'

        try:
            text, _ = self._complete_with_retry(
                lambda: self._load_client().complete_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    generation_params={"temperature": 0.0, "max_tokens": 300},
                )
            )
            events = _extract_json(text).get("events", [])
            parsed = [str(e).strip() for e in events if str(e).strip()][:max_events]
            self._record_success()
            return parsed
        except Exception as exc:  # noqa: BLE001 - temporal extraction is optional
            reported_exc = exc.cause if isinstance(exc, _LlmCallError) else exc
            self._record_failure(half_open=half_open)
            LOGGER.warning("Temporal event extraction failed: %s", reported_exc)
            return []

    def expand_query(self, visual_prompt: str, num_expansions: int = 4) -> list[str]:
        """Generate visually-descriptive English query variants for search recall.

        Ported from the AIC_2025 reference project's
        ``GPT4o_service.expand_query``. The first variant is always a plain
        translation of ``visual_prompt``; the rest vary visual perspective or
        description style without inventing new objects/actions. Returns an
        empty list (no expansion) when the LLM is unavailable.
        """
        if not visual_prompt.strip():
            return []

        can_call, half_open = self._acquire_circuit()
        if not can_call:
            return []

        system_prompt = (
            "You are an expert in search query optimization for a visual and "
            "video retrieval system. Expand the user's query into exactly "
            f"{num_expansions} diverse, visually descriptive English queries.\n"
            "Rules:\n"
            "1. The VERY FIRST query must be the simple, direct English "
            "translation/rephrasing of the original query.\n"
            "2. Stay true to the original: don't add objects, actions, or "
            "details not present in it. Only vary visual perspective, "
            "setting, or description style.\n"
            "3. Short, descriptive captions focusing on different visual "
            "aspects of the same core subject/action.\n"
            "4. Avoid abstract concepts or categorical lists.\n"
            'Return strict JSON: {"queries": ["query 1", "query 2", ...]} '
            "with no extra text."
        )
        user_prompt = f'Original query:\n"""{visual_prompt}"""\nReturn JSON now:'

        try:
            text, _ = self._complete_with_retry(
                lambda: self._load_client().complete_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    generation_params={"temperature": 0.3, "max_tokens": 400},
                )
            )
            queries = _extract_json(text).get("queries", [])
            parsed = [str(q).strip() for q in queries if str(q).strip()][:num_expansions]
            self._record_success()
            return parsed
        except Exception as exc:  # noqa: BLE001 - query expansion is optional
            reported_exc = exc.cause if isinstance(exc, _LlmCallError) else exc
            self._record_failure(half_open=half_open)
            LOGGER.warning("Query expansion failed: %s", reported_exc)
            return []


_OCR_SIGNALS: tuple[tuple[str, float], ...] = (
    ("dòng chữ", 0.95),
    ("dong chu", 0.95),
    ("biển hiệu", 0.95),
    ("bien hieu", 0.95),
    ("bảng hiệu", 0.95),
    ("bang hieu", 0.95),
    ("phụ đề", 0.9),
    ("phu de", 0.9),
    ("số điện thoại", 0.95),
    ("so dien thoai", 0.95),
    ("biển số", 0.95),
    ("bien so", 0.95),
    ("mã qr", 0.95),
    ("ma qr", 0.95),
    ("trên màn hình", 0.85),
    ("tren man hinh", 0.85),
    ("văn bản", 0.85),
    ("van ban", 0.85),
    ("quảng cáo", 0.8),
    ("quang cao", 0.8),
    ("nhãn hiệu", 0.85),
    ("nhan hieu", 0.85),
    ("logo", 0.9),
    ("chữ", 0.85),
    ("chu", 0.85),
    ("text", 0.8),
    ("sign", 0.8),
)

_ASR_SIGNALS: tuple[tuple[str, float], ...] = (
    ("phát biểu", 0.95),
    ("phat bieu", 0.95),
    ("lời thoại", 0.95),
    ("loi thoai", 0.95),
    ("nghe thấy", 0.95),
    ("nghe thay", 0.95),
    ("giọng nói", 0.9),
    ("giong noi", 0.9),
    ("đề cập", 0.9),
    ("de cap", 0.9),
    ("thảo luận", 0.9),
    ("thao luan", 0.9),
    ("phỏng vấn", 0.9),
    ("phong van", 0.9),
    ("nói về", 0.95),
    ("noi ve", 0.95),
    ("nói", 0.85),
    ("noi", 0.85),
    ("spoken", 0.85),
    ("speech", 0.85),
    ("says", 0.85),
)


def _heuristic_processed_query(query: str) -> ProcessedQuery:
    """Build a deterministic route for obvious Vietnamese/English modality cues."""
    ocr_confidence = _signal_confidence(query, _OCR_SIGNALS)
    asr_confidence = _signal_confidence(query, _ASR_SIGNALS)

    ocr_keywords = _heuristic_keywords(query, "ocr") if ocr_confidence else []
    asr_keywords = _heuristic_keywords(query, "asr") if asr_confidence else []
    confidence = {
        "kis": 1.0,
        "ocr": ocr_confidence,
        "asr": asr_confidence,
    }
    keyword_map = {
        "ocr": ocr_keywords,
        "asr": asr_keywords,
    }
    return ProcessedQuery(
        raw_query=query,
        visual_prompt=query,
        visual_prompt_vi=query,
        ocr_keywords=ocr_keywords,
        asr_keywords=asr_keywords,
        weights=_derive_bonus_weights(confidence, keyword_map),
        routing_mode="heuristic",
        modality_confidence=confidence,
        llm_status="not_requested",
        normalized_keywords=_normalize_keyword_map(keyword_map),
    )


def _merge_llm_and_heuristic(
    *,
    query: str,
    data: dict,
    heuristic: ProcessedQuery,
    llm_calls: int,
    llm_usage: dict[str, object] | None = None,
) -> ProcessedQuery:
    """Validate LLM fields and retain deterministic routes for unambiguous cues.

    ``data`` follows AIC_2025's ``analyze_intelligent_query`` output shape:
    ``{"components": {"kis"/"kis_vi"/"ocr"/"asr"}, "weights": {...}, "reasoning": "..."}``.
    OCR/ASR components are single strings there (a phrase, not a keyword
    list); each is wrapped as a one-element list to fit ``ProcessedQuery``'s
    keyword-list fields.
    """
    components = data.get("components")
    components = components if isinstance(components, dict) else {}

    llm_keywords = {
        "ocr": _wrap_component(components.get("ocr")),
        "asr": _wrap_component(components.get("asr")),
    }
    heuristic_keywords = {
        "ocr": heuristic.ocr_keywords,
        "asr": heuristic.asr_keywords,
    }
    keyword_map = {
        # Deterministic signal phrases take priority if the LLM already
        # returned the maximum number of candidates.
        modality: _deduplicate([*heuristic_keywords[modality], *llm_keywords[modality]])[:5]
        for modality in ("ocr", "asr")
    }

    raw_weights = data.get("weights")
    parsed_confidence = _parse_modality_confidence(raw_weights, llm_keywords)
    confidence = {
        "kis": max(0.5, parsed_confidence["kis"]),
        "ocr": max(parsed_confidence["ocr"], heuristic.modality_confidence["ocr"]),
        "asr": max(parsed_confidence["asr"], heuristic.modality_confidence["asr"]),
    }
    for modality in ("ocr", "asr"):
        if not keyword_map[modality]:
            confidence[modality] = 0.0

    visual_prompt = _nonempty_string(components.get("kis"), query)
    return ProcessedQuery(
        raw_query=query,
        visual_prompt=visual_prompt,
        visual_prompt_vi=_nonempty_string(components.get("kis_vi"), query),
        ocr_keywords=keyword_map["ocr"],
        asr_keywords=keyword_map["asr"],
        weights=_derive_bonus_weights(confidence, keyword_map),
        routing_mode="llm",
        modality_confidence=confidence,
        llm_status="ok",
        normalized_keywords=_normalize_keyword_map(keyword_map),
        llm_calls=llm_calls,
        llm_attempts=llm_calls,
        llm_usage=dict(llm_usage or {}),
    )


def _wrap_component(raw: object) -> list[str]:
    """Wrap a single OCR/ASR component string (AIC_2025's shape) as a keyword list."""
    if not isinstance(raw, str):
        return []
    value = " ".join(raw.split()).strip()
    return [value] if value else []


def _with_fallback(processed: ProcessedQuery, *, llm_status: str, reason: str) -> ProcessedQuery:
    processed.routing_mode = "fallback"
    processed.llm_status = llm_status
    processed.fallback_reason = reason
    return processed


def _nonempty_string(value: object, default: str) -> str:
    if not isinstance(value, str):
        return default
    value = value.strip()
    return value or default


def _parse_modality_confidence(raw: object, keyword_map: dict[str, list[str]]) -> dict[str, float]:
    raw = raw if isinstance(raw, dict) else {}
    result = {"kis": _clamp_confidence(raw.get("kis"), default=1.0)}
    for modality in ("ocr", "asr"):
        default = 0.65 if keyword_map.get(modality) else 0.0
        value = _clamp_confidence(raw.get(modality), default=default)
        result[modality] = value if keyword_map.get(modality) else 0.0
    return result


def _clamp_confidence(value: object, *, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _derive_bonus_weights(
    confidence: dict[str, float], keyword_map: dict[str, list[str]]
) -> dict[str, float]:
    """Turn routing confidence into bounded bonuses; raw LLM weights are ignored."""
    return {
        f"{modality}_bonus": (
            round(min(0.5, max(0.0, confidence.get(modality, 0.0)) * 0.5), 6)
            if keyword_map.get(modality)
            else 0.0
        )
        for modality in ("ocr", "asr")
    }


def _signal_confidence(query: str, signals: tuple[tuple[str, float], ...]) -> float:
    folded = unicodedata.normalize("NFKC", query).casefold()
    return max(
        (
            confidence
            for signal, confidence in signals
            if re.search(
                rf"(?<!\w){re.escape(unicodedata.normalize('NFKC', signal).casefold())}(?!\w)",
                folded,
            )
        ),
        default=0.0,
    )


def _heuristic_keywords(query: str, modality: str) -> list[str]:
    """Extract a useful phrase after a modality cue while retaining the original wording."""
    query = " ".join(query.split()).strip()
    if not query:
        return []

    quoted = [m.group(1).strip() for m in re.finditer(r'["“”]([^"“”]+)["“”]', query)]
    if modality == "ocr" and quoted:
        return _deduplicate(quoted)[:5]

    cue_patterns = {
        "ocr": (
            r"dòng chữ|dong chu|biển hiệu|bien hieu|bảng hiệu|bang hieu|phụ đề|phu de|"
            r"số điện thoại|so dien thoai|biển số|bien so|mã qr|ma qr|logo|chữ|chu|text|sign"
            r"|quảng cáo|quang cao|nhãn hiệu|nhan hieu"
        ),
        "asr": (
            r"phát biểu|phat bieu|lời thoại|loi thoai|nghe thấy|nghe thay|giọng nói|"
            r"giong noi|đề cập|de cap|thảo luận|thao luan|phỏng vấn|phong van|"
            r"nói về|noi ve|nói|noi|spoken|speech|says"
        ),
    }
    if modality in cue_patterns:
        bounded_pattern = rf"(?<!\w)(?:{cue_patterns[modality]})(?!\w)"
        matches = list(re.finditer(bounded_pattern, query, flags=re.IGNORECASE))
        if matches:
            matched_cue = matches[-1].group(0).strip()
            candidate = query[matches[-1].end() :]
            candidate = re.split(r"\s*[,;]\s*", candidate, maxsplit=1)[0]
            candidate = re.sub(
                r"^\s*(?:về|ve|rằng|rang|là|la|ghi|viết|viet|hiển thị|hien thi)\s*",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
            if modality == "ocr":
                candidate = re.sub(
                    r"\s*(?:trên|tren)\s+(?:màn hình|man hinh|biển|bien)\s*$",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                )
            candidate = candidate.strip(" ,.;:!?-")
            if candidate:
                return [candidate]
            if modality == "ocr":
                return [matched_cue]

    candidate = re.sub(
        r"^\s*(?:hãy\s+|hay\s+)?(?:tìm|tim|tìm kiếm|tim kiem|cho tôi xem|cho toi xem)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    return [candidate or query]


def _normalize_keyword_map(keyword_map: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        modality: _normalize_keywords(keyword_map.get(modality, [])) for modality in ("ocr", "asr")
    }


def _normalize_keywords(keywords: list[str]) -> list[str]:
    normalized: list[str] = []
    for keyword in keywords:
        lowercase = unicodedata.normalize("NFKC", keyword).casefold()
        lowercase = " ".join(lowercase.split())
        if lowercase:
            normalized.append(lowercase)
            unaccented = _remove_diacritics(lowercase)
            if unaccented != lowercase:
                normalized.append(unaccented)
    return _deduplicate(normalized)


def _remove_diacritics(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        import httpx

        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429 or exc.response.status_code >= 500
    except ImportError:  # pragma: no cover - httpx is present with the vLLM client
        pass
    return False


def _extract_json(text: str) -> dict:
    """Parse the first JSON object out of the LLM response (may be fenced/prefixed)."""
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in LLM response: {text[:120]!r}")
    data = json.loads(match.group())
    if not isinstance(data, dict):
        raise TypeError("LLM response JSON is not an object")
    return data


def get_query_processor() -> QueryProcessor:
    """Return the LLM query processor over OpenRouter.

    The processor auto-degrades to heuristic routing when OpenRouter isn't
    reachable or misconfigured, so returning it unconditionally is safe.
    """
    return LlmQueryProcessor()
