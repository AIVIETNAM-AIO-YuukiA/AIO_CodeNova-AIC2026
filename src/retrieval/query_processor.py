"""Query processor components for translation, slot filling, and expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os

LOGGER = logging.getLogger(__name__)

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
    # Bonus multipliers for auxiliary text modalities (typically 0.0 to 0.3).
    # Populated by LlmQueryProcessor. Visual KIS is implicitly the base (1.0).
    weights: dict[str, float] = field(
        default_factory=lambda: {"ocr_bonus": 0.0, "asr_bonus": 0.0, "caption_bonus": 0.0}
    )


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

    def process(
        self, query: str, enabled_models: list[str] | None = None, use_llm: bool = True
    ) -> ProcessedQuery:
        if self._disabled or not use_llm:
            return ProcessedQuery(raw_query=query, visual_prompt=query, visual_prompt_vi=query)

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
        7. "weights": A JSON object {{"caption_bonus": float, "ocr_bonus": float, "asr_bonus": float}}, estimating the bonus multiplier for each modality (from 0.0 to 0.3). Visual KIS is the main component and implicitly has weight 1.0. A modality with an empty keyword list should get bonus 0.0.

        User Query: "{query}"

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
            "weights": {{"caption_bonus": 0.0, "ocr_bonus": 0.0, "asr_bonus": 0.0}}
        }}
        """
        try:
            text = self._load_client().complete_text(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                generation_params={"temperature": 0.0, "max_tokens": 400},
            )
            data = _extract_json(text)
            LOGGER.info("Qwen Preprocessing Result: %s", json.dumps(data, ensure_ascii=False))
            weights = data.get("weights") or {}
            return ProcessedQuery(
                raw_query=query,
                visual_prompt=data.get("visual_prompt", query) or query,
                visual_prompt_vi=data.get("visual_prompt_vi", query) or query,
                caption_keywords=data.get("caption_keywords", []),
                ocr_keywords=data.get("ocr_keywords", []),
                asr_keywords=data.get("asr_keywords", []),
                metadata=data.get("metadata", {}),
                weights=_parse_bonus_weights(weights),
            )
        except Exception as exc:
            LOGGER.warning(
                "LLM query processing failed: %s. Falling back to pass-through "
                "for the rest of this session.",
                exc,
            )
            self._disabled = True
            return ProcessedQuery(raw_query=query, visual_prompt=query, visual_prompt_vi=query)

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
        raise ValueError("LLM response JSON is not an object")
    return data


def get_query_processor() -> QueryProcessor:
    """Return the LLM query processor over the Docker agent endpoint.

    No API key needed — the processor auto-degrades to pass-through when the
    endpoint at ``AGENT_LOCAL_ENGINE_URL`` isn't reachable, so returning it
    unconditionally is safe.
    """
    return LlmQueryProcessor()
