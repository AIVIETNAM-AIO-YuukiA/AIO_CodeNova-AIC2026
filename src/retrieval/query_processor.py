"""Query processor components for translation, slot filling, and expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os

LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT = "You are an expert query processor for a video retrieval system."


@dataclass
class ProcessedQuery:
    """Structured representation of a processed search query."""

    raw_query: str
    visual_prompt: str
    ocr_keywords: list[str] = field(default_factory=list)
    asr_keywords: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    # How much each modality should contribute to a fused multi-source search
    # (kis/ocr/asr, sums to ~1.0). Populated by LlmQueryProcessor; a
    # pass-through processor leaves this at the all-visual default.
    weights: dict[str, float] = field(default_factory=lambda: {"kis": 1.0, "ocr": 0.0, "asr": 0.0})


class QueryProcessor:
    """Base interface for query processing."""

    def process(self, query: str) -> ProcessedQuery:
        """Process a raw user query into structured fields."""
        raise NotImplementedError

    def extract_temporal_events(self, query: str, max_events: int = 5) -> list[str]:
        """Split a query into ordered events; empty list means "no sequence"."""
        return []


class PassThroughQueryProcessor(QueryProcessor):
    """Query processor that passes the raw query directly without modifications."""

    def process(self, query: str) -> ProcessedQuery:
        return ProcessedQuery(raw_query=query, visual_prompt=query)


class LlmQueryProcessor(QueryProcessor):
    """Translate/expand queries via the Docker-hosted agent LLM (port 8888).

    Falls back silently to pass-through on any server/parse error and disables
    itself after the first connection failure — a competition run must never
    hard-fail (or pay a timeout on every query) because the LLM container
    isn't up.
    """

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        self._base_url = base_url
        self._model_name = model_name
        self._client = None
        self._disabled = False

    def _load_client(self):
        if self._client is None:
            from modules._vllm_chat import VllmChatClient

            self._client = VllmChatClient(
                base_url=self._base_url
                or os.environ.get("AGENT_LOCAL_ENGINE_URL", "http://localhost:8884/v1"),
                model_name=self._model_name
                or os.environ.get("AGENT_LOCAL_ENGINE_MODEL", "cyankiwi/Qwen3.5-4B-AWQ-4bit"),
            )
        return self._client

    def process(self, query: str) -> ProcessedQuery:
        if self._disabled:
            return ProcessedQuery(raw_query=query, visual_prompt=query)

        prompt = f"""
        Analyze the user query (which might be in Vietnamese or English) and output a JSON object with the following fields:

        1. "visual_prompt": If the query is in Vietnamese, translate it to English. Then rewrite it as a natural, easy-to-understand visual search prompt for vision-language video retrieval. Stay as close as possible to the user's original query. Preserve all entities and intent exactly. Do not add, infer, or invent any actions, objects, people, locations, events, attributes, camera details, lighting, colors, or other visual elements that are not explicitly mentioned in the query. Only rephrase for clarity and naturalness. Keep the output concise (max 50 words).
        2. "ocr_keywords": List 1 to 5 search keywords (in English and Vietnamese if applicable) representing text, signs, logos, or writing that might appear *on screen* (OCR text). Set to empty list if no text/signs are implied.
        3. "asr_keywords": List 1 to 5 search keywords (in English and Vietnamese if applicable) representing words or topics that might be *spoken* (ASR speech). Set to empty list if no speech/dialogue is implied.
        4. "metadata": A JSON dictionary of extracted attributes like "color", "weather", "time_of_day", "location_type" (indoor/outdoor), or "action", if explicitly mentioned.
        5. "weights": A JSON object {{"kis": float, "ocr": float, "asr": float}} summing to 1.0, estimating how much each modality should contribute to retrieval. Visual-dominant queries (actions, scenes, objects) weigh "kis" highest. Queries naming exact on-screen text/numbers weigh "ocr" highest. Queries about spoken topics/dialogue weigh "asr" highest. A modality with an empty keyword list gets weight 0.

        User Query: "{query}"

        Respond with ONLY the JSON object:
        {{
            "visual_prompt": "string",
            "ocr_keywords": ["string"],
            "asr_keywords": ["string"],
            "metadata": {{
                "key": "value"
            }},
            "weights": {{"kis": 1.0, "ocr": 0.0, "asr": 0.0}}
        }}
        """
        try:
            text = self._load_client().complete_text(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                generation_params={"temperature": 0.0, "max_tokens": 400},
            )
            data = _extract_json(text)
            weights = data.get("weights") or {}
            return ProcessedQuery(
                raw_query=query,
                visual_prompt=data.get("visual_prompt", query) or query,
                ocr_keywords=data.get("ocr_keywords", []),
                asr_keywords=data.get("asr_keywords", []),
                metadata=data.get("metadata", {}),
                weights=_normalize_weights(weights),
            )
        except Exception as exc:
            LOGGER.warning(
                "LLM query processing failed: %s. Falling back to pass-through "
                "for the rest of this session.",
                exc,
            )
            self._disabled = True
            return ProcessedQuery(raw_query=query, visual_prompt=query)

    def extract_temporal_events(self, query: str, max_events: int = 5) -> list[str]:
        """Split one free-form query into an ordered list of atomic events.

        Lets a user type "a man walks out, then rides a motorbike, then turns
        left" instead of filling in the TRAKE event boxes by hand. Returns an
        empty list when the LLM is unavailable or the query holds no sequence,
        so callers can fall back to treating the query as a single event.

        Ported from the AIC_2025 reference project's
        ``GPT4o_service.extract_temporal_events``.
        """
        if self._disabled or not query.strip():
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
            text = self._load_client().complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_params={"temperature": 0.0, "max_tokens": 300},
            )
            events = _extract_json(text).get("events", [])
            return [str(e).strip() for e in events if str(e).strip()][:max_events]
        except Exception as exc:
            LOGGER.warning("Temporal event extraction failed: %s", exc)
            return []


def _normalize_weights(raw: dict) -> dict[str, float]:
    """Coerce an LLM-provided weight dict to non-negative floats summing to 1.0.

    Falls back to an all-visual split if the response omits weights, uses
    unrecognized keys, or sums to (near) zero.
    """
    weights = {}
    for key in ("kis", "ocr", "asr"):
        try:
            weights[key] = max(0.0, float(raw.get(key, 0.0)))
        except (TypeError, ValueError):
            weights[key] = 0.0

    total = sum(weights.values())
    if total <= 1e-9:
        return {"kis": 1.0, "ocr": 0.0, "asr": 0.0}
    return {key: value / total for key, value in weights.items()}


def _extract_json(text: str) -> dict:
    """Parse the first JSON object out of the LLM response (may be fenced/prefixed)."""
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in LLM response: {text[:120]!r}")
    data = json.loads(match.group())
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON is not an object")
    return data


def get_query_processor() -> QueryProcessor:
    """Return the LLM query processor over the Docker agent endpoint.

    No API key needed — the processor auto-degrades to pass-through when the
    endpoint at ``AGENT_LOCAL_ENGINE_URL`` isn't reachable, so returning it
    unconditionally is safe.
    """
    return LlmQueryProcessor()
