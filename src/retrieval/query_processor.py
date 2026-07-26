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


class QueryProcessor:
    """Base interface for query processing."""

    def process(self, query: str) -> ProcessedQuery:
        """Process a raw user query into structured fields."""
        raise NotImplementedError


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
            from agent.hardware import default_agent_model
            from modules._vllm_chat import VllmChatClient

            self._client = VllmChatClient(
                base_url=self._base_url
                or os.environ.get("AGENT_LOCAL_ENGINE_URL", "http://localhost:8888/v1"),
                model_name=self._model_name or default_agent_model(),
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

        User Query: "{query}"

        Respond with ONLY the JSON object:
        {{
            "visual_prompt": "string",
            "ocr_keywords": ["string"],
            "asr_keywords": ["string"],
            "metadata": {{
                "key": "value"
            }}
        }}
        """
        try:
            text = self._load_client().complete_text(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                generation_params={"temperature": 0.0, "max_tokens": 400},
            )
            data = _extract_json(text)
            return ProcessedQuery(
                raw_query=query,
                visual_prompt=data.get("visual_prompt", query) or query,
                ocr_keywords=data.get("ocr_keywords", []),
                asr_keywords=data.get("asr_keywords", []),
                metadata=data.get("metadata", {}),
            )
        except Exception as exc:
            LOGGER.warning(
                "LLM query processing failed: %s. Falling back to pass-through "
                "for the rest of this session.",
                exc,
            )
            self._disabled = True
            return ProcessedQuery(raw_query=query, visual_prompt=query)


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
    server (``make agent-up``) isn't running, so returning it unconditionally
    is safe.
    """
    return LlmQueryProcessor()
