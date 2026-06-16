"""Query processor components for translation, slot filling, and expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os

LOGGER = logging.getLogger(__name__)


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
    """Query processor backed by Gemini LLM to translate, expand, and parse queries."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._client = None
        self._initialized = False

    def _init_client(self) -> bool:
        if self._initialized:
            return True
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._initialized = True
            return True
        except Exception as exc:
            LOGGER.exception("Failed to initialize Gemini client: %s", exc)
            return False

    def process(self, query: str) -> ProcessedQuery:
        if not self._init_client():
            LOGGER.warning("Gemini client not initialized, falling back to pass-through.")
            return ProcessedQuery(raw_query=query, visual_prompt=query)

        prompt = f"""
        You are an expert query processor for a video retrieval system.
        Your task is to analyze the user query (which might be in Vietnamese or English) and output a JSON object with the following fields:
        
        1. "visual_prompt": Translate the query to English (if it's in Vietnamese) and enrich it with descriptive visual details (e.g., typical scene composition, objects, colors, camera angles/shots, lighting) that would help a CLIP (Contrastive Language-Image Pre-training) model find relevant video keyframes. Be descriptive but stay true to the original intent.
        2. "ocr_keywords": List 1 to 5 search keywords (in English and Vietnamese if applicable) representing text, signs, logos, or writing that might appear *on screen* (OCR text). Set to empty list if no text/signs are implied.
        3. "asr_keywords": List 1 to 5 search keywords (in English and Vietnamese if applicable) representing words or topics that might be *spoken* (ASR speech). Set to empty list if no speech/dialogue is implied.
        4. "metadata": A JSON dictionary of extracted attributes like "color", "weather", "time_of_day", "location_type" (indoor/outdoor), or "action", if explicitly mentioned.
        
        User Query: "{query}"
        
        Output JSON format:
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
            from google.genai import types
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text.strip())
            return ProcessedQuery(
                raw_query=query,
                visual_prompt=data.get("visual_prompt", query),
                ocr_keywords=data.get("ocr_keywords", []),
                asr_keywords=data.get("asr_keywords", []),
                metadata=data.get("metadata", {}),
            )
        except Exception as exc:
            LOGGER.warning("Gemini query processing failed: %s. Falling back to pass-through.", exc)
            return ProcessedQuery(raw_query=query, visual_prompt=query)


def get_query_processor() -> QueryProcessor:
    """Resolve and return the appropriate query processor based on environment configuration."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        LOGGER.info("GEMINI_API_KEY environment variable not set. Using PassThroughQueryProcessor.")
        return PassThroughQueryProcessor()
    
    LOGGER.info("GEMINI_API_KEY found. Using LlmQueryProcessor.")
    return LlmQueryProcessor(api_key=api_key)