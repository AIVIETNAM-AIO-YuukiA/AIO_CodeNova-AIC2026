"""Query expansion / decomposition prompts (stub)."""

from __future__ import annotations

_TEMPLATE = """You are a video retrieval query assistant.
Rewrite the user query into {n} diverse English search variants that describe
the same visual moment. Return one variant per line.

Query: {query}
"""


def build_query_expansion_prompt(query: str, n: int = 4) -> str:
    """Return a prompt asking an LLM for ``n`` query variants."""
    return _TEMPLATE.format(query=query.strip(), n=n)
