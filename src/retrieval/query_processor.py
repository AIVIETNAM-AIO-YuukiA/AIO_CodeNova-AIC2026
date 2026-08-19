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


_SYSTEM_PROMPT = """You are an expert AI query optimization assistant for Visual Information Retrieval systems (like SigLIP, CLIP). Your sole purpose is to convert natural language queries into highly optimized, static, and purely visual English queries.

THE CORE PRINCIPLE
Visual embedding models DO NOT understand abstract concepts, emotions, intentions, cultural contexts, or dynamic actions over time. They ONLY understand what can be seen in a single freeze-frame: objects, colors, spatial layouts, and physical shapes.

Your task is to strip away everything invisible and translate the remaining visual elements into English.

RULE 1: Action-to-Static Conversion
Rewrite any action/verb phrase so the result describes what a single freeze-frame would show (e.g., "mọi người nhảy múa" -> "people on a stage", "người đang cầu nguyện" -> "a person sitting").

RULE 2: The Critical Filter (Remove Unsearchable Elements)
If a detail cannot be clearly represented by a specific object, color, or shape in a static frame, YOU MUST REMOVE IT entirely. DO NOT include:
1. Abstract Behaviors (e.g., cầu nguyện, khấn vái -> keep only the physical posture).
2. Purposes & Intentions (e.g., cầu cho chuyến đi bình an -> remove).
3. Emotions (e.g., vui mừng, xúc động -> remove, unless a clear physical expression like smiling).
4. Symbolic Meanings (remove).
5. Cultural & Historical Context (e.g., lễ hội Obon -> remove proper nouns, keep "people in traditional costumes").
6. Social Relationships (e.g., người dân, cụ già -> keep only visual identifiers like "elderly person")."""


@dataclass
class ProcessedQuery:
    """Structured representation of a processed search query."""

    raw_query: str
    visual_prompt: str  # English visual prompt
    visual_prompt_vi: str = ""  # Vietnamese visual prompt
    caption_keywords: list[str] = field(default_factory=list)
    ocr_keywords: list[str] = field(default_factory=list)
    asr_keywords: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    # Bonus multipliers derived from routing confidence for backward-compatible
    # retrieval callers. Visual KIS is implicitly the base (1.0).
    weights: dict[str, float] = field(
        default_factory=lambda: {"ocr_bonus": 0.0, "asr_bonus": 0.0, "caption_bonus": 0.0}
    )
    # Additive routing/observability fields.  The original fields above remain
    # unchanged so older retrieval callers can continue to consume this type.
    routing_mode: str = "heuristic"  # llm | heuristic | fallback
    modality_confidence: dict[str, float] = field(
        default_factory=lambda: {"kis": 1.0, "ocr": 0.0, "asr": 0.0, "caption": 0.0}
    )
    llm_status: str = "not_requested"  # ok | disabled | error | circuit_open
    fallback_reason: str | None = None
    normalized_keywords: dict[str, list[str]] = field(
        default_factory=lambda: {"caption": [], "ocr": [], "asr": []}
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


class PassThroughQueryProcessor(QueryProcessor):
    """Query processor that passes the raw query directly without modifications."""

    def process(
        self, query: str, enabled_models: list[str] | None = None, use_llm: bool = True
    ) -> ProcessedQuery:
        return ProcessedQuery(raw_query=query, visual_prompt=query, visual_prompt_vi=query)


class LlmQueryProcessor(QueryProcessor):
    """Translate/expand queries via the Docker-hosted agent LLM (port 8888).

    The LLM augments deterministic routing rather than being a single point of
    failure.  A failed request falls back to Vietnamese/English heuristics for
    that request.  A small circuit breaker prevents repeated timeouts while
    still probing the backend again after a cooldown.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        *,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 60.0,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._base_url = base_url
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
                base_url=self._base_url
                or os.environ.get("AGENT_LOCAL_ENGINE_URL", "http://localhost:8884/v1"),
                model_name=self._model_name
                or os.environ.get("AGENT_LOCAL_ENGINE_MODEL", "cyankiwi/Qwen3.5-4B-AWQ-4bit"),
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

        prompt = f"""
        Analyze the user query (which might be in Vietnamese or English) and output a JSON object with the following fields:

        1. "visual_prompt": Apply RULE 1 and RULE 2 from the system prompt to strictly filter out all invisible elements, abstract behaviors, purposes, emotions, symbols, and cultural context. Translate the remaining static visual essence into a highly optimized English noun-phrase. Keep it concise (max 20 words). DO NOT include verbs of abstract actions.
        { '2. "visual_prompt_vi": The exact same visual essence as "visual_prompt", but translated back into Vietnamese (static noun-phrase in Vietnamese).' if is_vi else '' }
        3. "caption_keywords": List 1 to 5 search keywords describing the static scene, objects, or context that would appear in a textual image caption. Set to empty list if no clear context is given.
        4. "ocr_keywords": List 1 to 5 search keywords (in English and Vietnamese if applicable) representing text, signs, logos, or writing that might appear *on screen* (OCR text). Set to empty list if no text/signs are implied.
        5. "asr_keywords": List 1 to 5 search keywords (in English and Vietnamese if applicable) representing words or topics that might be *spoken* (ASR speech). Set to empty list if no speech/dialogue is implied.
        6. "metadata": A JSON dictionary of extracted attributes like "color", "weather", "time_of_day", "location_type" (indoor/outdoor).
        7. "modality_confidence": A JSON object {{"kis": float, "caption": float, "ocr": float, "asr": float}}. Each value is a confidence from 0.0 to 1.0 that the modality is useful for this query. Use 0.0 when its keyword list is empty. KIS should normally remain enabled.

        User Query: {json.dumps(query, ensure_ascii=False)}

        Respond with ONLY the JSON object:
        {{
            "visual_prompt": "string",
            { '"visual_prompt_vi": "string",' if is_vi else '' }
            "caption_keywords": ["string"],
            "ocr_keywords": ["string"],
            "asr_keywords": ["string"],
            "metadata": {{
                "key": "value"
            }},
            "modality_confidence": {{"kis": 1.0, "caption": 0.0, "ocr": 0.0, "asr": 0.0}}
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

_CAPTION_SIGNALS: tuple[tuple[str, float], ...] = (
    ("bối cảnh", 0.65),
    ("boi canh", 0.65),
    ("khung cảnh", 0.65),
    ("khung canh", 0.65),
    ("mô tả", 0.65),
    ("mo ta", 0.65),
    ("cảnh có", 0.55),
    ("canh co", 0.55),
    ("trong nhà", 0.55),
    ("trong nha", 0.55),
    ("ngoài trời", 0.55),
    ("ngoai troi", 0.55),
    ("người", 0.45),
    ("nguoi", 0.45),
    ("vật thể", 0.45),
    ("vat the", 0.45),
    ("đồ vật", 0.45),
    ("do vat", 0.45),
    ("động vật", 0.45),
    ("dong vat", 0.45),
    ("xe", 0.45),
    ("ô tô", 0.45),
    ("o to", 0.45),
    ("đang", 0.45),
    ("cầm", 0.45),
    ("cam", 0.45),
    ("mặc", 0.45),
    ("mac", 0.45),
    ("đứng", 0.45),
    ("dung", 0.45),
    ("ngồi", 0.45),
    ("ngoi", 0.45),
    ("chạy", 0.45),
    ("chay", 0.45),
    ("scene", 0.55),
    ("person", 0.45),
    ("object", 0.45),
    ("animal", 0.45),
    ("car", 0.45),
)


def _heuristic_processed_query(query: str) -> ProcessedQuery:
    """Build a deterministic route for obvious Vietnamese/English modality cues."""
    ocr_confidence = _signal_confidence(query, _OCR_SIGNALS)
    asr_confidence = _signal_confidence(query, _ASR_SIGNALS)
    caption_confidence = _signal_confidence(query, _CAPTION_SIGNALS)

    ocr_keywords = _heuristic_keywords(query, "ocr") if ocr_confidence else []
    asr_keywords = _heuristic_keywords(query, "asr") if asr_confidence else []
    caption_keywords = _heuristic_keywords(query, "caption") if caption_confidence else []
    confidence = {
        "kis": 1.0,
        "ocr": ocr_confidence,
        "asr": asr_confidence,
        "caption": caption_confidence,
    }
    keyword_map = {
        "caption": caption_keywords,
        "ocr": ocr_keywords,
        "asr": asr_keywords,
    }
    return ProcessedQuery(
        raw_query=query,
        visual_prompt=query,
        visual_prompt_vi=query,
        caption_keywords=caption_keywords,
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
    """Validate LLM fields and retain deterministic routes for unambiguous cues."""
    llm_keywords = {
        "caption": _parse_keywords(data.get("caption_keywords")),
        "ocr": _parse_keywords(data.get("ocr_keywords")),
        "asr": _parse_keywords(data.get("asr_keywords")),
    }
    heuristic_keywords = {
        "caption": heuristic.caption_keywords,
        "ocr": heuristic.ocr_keywords,
        "asr": heuristic.asr_keywords,
    }
    keyword_map = {
        # Deterministic signal phrases take priority if the LLM already
        # returned the maximum number of candidates.
        modality: _deduplicate([*heuristic_keywords[modality], *llm_keywords[modality]])[:5]
        for modality in ("caption", "ocr", "asr")
    }

    raw_confidence = data.get("modality_confidence")
    if not isinstance(raw_confidence, dict):
        # Accept the shorter name from early prompt experiments, but never use
        # legacy raw bonus weights to decide fusion weights.
        raw_confidence = data.get("confidence")
    parsed_confidence = _parse_modality_confidence(raw_confidence, llm_keywords)
    confidence = {
        "kis": max(0.5, parsed_confidence["kis"]),
        "caption": max(parsed_confidence["caption"], heuristic.modality_confidence["caption"]),
        "ocr": max(parsed_confidence["ocr"], heuristic.modality_confidence["ocr"]),
        "asr": max(parsed_confidence["asr"], heuristic.modality_confidence["asr"]),
    }
    for modality in ("caption", "ocr", "asr"):
        if not keyword_map[modality]:
            confidence[modality] = 0.0

    raw_metadata = data.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    return ProcessedQuery(
        raw_query=query,
        visual_prompt=_nonempty_string(data.get("visual_prompt"), query),
        visual_prompt_vi=_nonempty_string(data.get("visual_prompt_vi"), query),
        caption_keywords=keyword_map["caption"],
        ocr_keywords=keyword_map["ocr"],
        asr_keywords=keyword_map["asr"],
        metadata=metadata,
        weights=_derive_bonus_weights(confidence, keyword_map),
        routing_mode="llm",
        modality_confidence=confidence,
        llm_status="ok",
        normalized_keywords=_normalize_keyword_map(keyword_map),
        llm_calls=llm_calls,
        llm_attempts=llm_calls,
        llm_usage=dict(llm_usage or {}),
    )


def _with_fallback(
    processed: ProcessedQuery, *, llm_status: str, reason: str
) -> ProcessedQuery:
    processed.routing_mode = "fallback"
    processed.llm_status = llm_status
    processed.fallback_reason = reason
    return processed


def _nonempty_string(value: object, default: str) -> str:
    if not isinstance(value, str):
        return default
    value = value.strip()
    return value or default


def _parse_keywords(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    values = []
    for value in raw:
        if not isinstance(value, str):
            continue
        value = " ".join(value.split()).strip()
        if value:
            values.append(value)
    return _deduplicate(values)[:5]


def _parse_modality_confidence(
    raw: object, keyword_map: dict[str, list[str]]
) -> dict[str, float]:
    raw = raw if isinstance(raw, dict) else {}
    result = {"kis": _clamp_confidence(raw.get("kis"), default=1.0)}
    for modality in ("caption", "ocr", "asr"):
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
        for modality in ("caption", "ocr", "asr")
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
        modality: _normalize_keywords(keyword_map.get(modality, []))
        for modality in ("caption", "ocr", "asr")
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


def _parse_bonus_weights(raw: dict) -> dict[str, float]:
    """Extract bonus multipliers for text modalities (capped at 0.5 to prevent blowing up the score)."""
    weights = {}
    for key in ("caption_bonus", "ocr_bonus", "asr_bonus"):
        try:
            weights[key] = max(0.0, min(0.5, float(raw.get(key, 0.0))))
        except (TypeError, ValueError):
            weights[key] = 0.0
    return weights


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
    """Return the LLM query processor over the Docker agent endpoint.

    No API key needed — the processor auto-degrades to pass-through when the
    endpoint at ``AGENT_LOCAL_ENGINE_URL`` isn't reachable, so returning it
    unconditionally is safe.
    """
    return LlmQueryProcessor()
