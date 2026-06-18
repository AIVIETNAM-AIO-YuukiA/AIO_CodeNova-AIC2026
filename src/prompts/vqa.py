"""Visual question answering prompts (stub)."""

from __future__ import annotations

_TEMPLATE = """Answer the question using only the visual evidence described below.
If the evidence is insufficient, say so.

Question: {question}
Evidence: {context}
"""


def build_vqa_prompt(question: str, context: str = "") -> str:
    """Return a VQA answer-generation prompt."""
    return _TEMPLATE.format(question=question.strip(), context=context.strip())
