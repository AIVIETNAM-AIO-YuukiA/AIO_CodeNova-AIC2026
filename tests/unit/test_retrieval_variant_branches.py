"""Unit coverage for high-recall per-model query branches."""

from __future__ import annotations

from types import SimpleNamespace

from core.types import SearchResult
from retrieval.search import Retriever


class _Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.queries: list[str] = []

    def embed_text(self, text: str):
        self.queries.append(text)
        return (self.model_name, text)


class _Index:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int, str | None]] = []

    def search(self, query_embedding, top_k: int = 300, model_name: str | None = None):
        self.calls.append((query_embedding, top_k, model_name))
        query = str(query_embedding[1]).replace(" ", "-")
        return [
            SearchResult(
                frame_id=f"{model_name}-{query}",
                video_id=f"video-{model_name}",
                score=1.0,
            )
        ]


class _Hydrator:
    def hydrate_with_diagnostics(self, results):
        return SimpleNamespace(results=list(results), issues=[])


def test_search_variant_branches_runs_each_language_through_each_model() -> None:
    jina = _Embedder("jina")
    siglip = _Embedder("siglip")
    index = _Index()
    retriever = Retriever(
        experiment=object(),
        embedders={"jina": jina, "siglip": siglip},
        index=index,
        hydrator=_Hydrator(),
        query_processor=object(),
    )

    branches = retriever.search_variant_branches(
        {"en": ["four ingredients"], "vi": ["bốn nguyên liệu"]},
        top_k=500,
    )

    assert set(branches) == {
        "jina:en:0",
        "jina:vi:0",
        "siglip:en:0",
        "siglip:vi:0",
    }
    assert jina.queries == ["four ingredients", "bốn nguyên liệu"]
    assert siglip.queries == ["four ingredients", "bốn nguyên liệu"]
    assert all(top_k == 500 for _, top_k, _ in index.calls)


def test_search_variant_branches_respects_enabled_models() -> None:
    jina = _Embedder("jina")
    siglip = _Embedder("siglip")
    index = _Index()
    retriever = Retriever(
        experiment=object(),
        embedders={"jina": jina, "siglip": siglip},
        index=index,
        hydrator=_Hydrator(),
        query_processor=object(),
    )

    branches = retriever.search_variant_branches(
        {"en": ["woman holding two ingredients"]},
        top_k=500,
        enabled_models=["siglip"],
    )

    assert set(branches) == {"siglip:en:0"}
    assert jina.queries == []
    assert siglip.queries == ["woman holding two ingredients"]
