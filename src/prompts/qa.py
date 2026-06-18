"""Question answering prompts (stub)."""

from __future__ import annotations

_TEMPLATE = """Answer the question concisely based on the provided context.

Question: {question}
Context: {context}
"""


def build_qa_prompt(question: str, context: str = "") -> str:
    """Return a QA answer-generation prompt."""
    return _TEMPLATE.format(question=question.strip(), context=context.strip())
