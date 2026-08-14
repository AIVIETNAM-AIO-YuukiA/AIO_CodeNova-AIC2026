from types import SimpleNamespace

import pytest

from config.settings import Experiment, PipelineConfig
from core.errors import RetrievalError
from core.types import SearchResult
from retrieval.hydrator import HydrationBatch
from retrieval.search import Retriever
from ui.server import _warmup_models


class _Embedder:
    def __init__(self, calls: list[str], name: str, *, fail: bool = False) -> None:
        self.calls = calls
        self.name = name
        self.fail = fail

    def embed_text(self, query: str):
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return [1.0]


class _Reranker:
    def __init__(self, *, fail_load: bool = False, fail_rerank: bool = False) -> None:
        self.fail_load = fail_load
        self.fail_rerank = fail_rerank
        self.rerank_calls = 0

    def _load(self):
        if self.fail_load:
            raise RuntimeError("load failed")

    def rerank(self, query: str, results: list[SearchResult]):
        self.rerank_calls += 1
        if self.fail_rerank:
            raise RuntimeError("inference failed")
        return list(reversed(results))


def _experiment(tmp_path):
    return Experiment("exp", tmp_path, PipelineConfig(runs_dir=tmp_path))


def test_all_embedders_are_checked_and_any_failure_blocks_activation(tmp_path):
    calls: list[str] = []
    retriever = SimpleNamespace(
        embedders={
            "one": _Embedder(calls, "one"),
            "two": _Embedder(calls, "two", fail=True),
            "three": _Embedder(calls, "three"),
        },
        reranker=None,
    )

    with pytest.raises(RetrievalError, match="two"):
        _warmup_models(None, _experiment(tmp_path), retriever)

    assert calls == ["one", "two", "three"]


def test_failed_optional_rerankers_are_disabled(tmp_path):
    retriever = SimpleNamespace(embedders={}, reranker=_Reranker(fail_load=True))

    report = _warmup_models(_Reranker(fail_load=True), _experiment(tmp_path), retriever)

    assert report.status == "DEGRADED"
    assert report.ui_reranker is None
    assert retriever.reranker is None
    assert {item.component: item.status for item in report.components} == {
        "reranker:ui": "FAILED",
        "reranker:retriever": "FAILED",
    }


def test_ui_reranker_falls_back_once_then_stays_disabled(tmp_path):
    raw = [SearchResult("f1", "v1", 1.0)]
    reranker = _Reranker(fail_rerank=True)
    report = _warmup_models(
        reranker,
        _experiment(tmp_path),
        SimpleNamespace(embedders={}, reranker=None),
    )

    assert report.ui_reranker.rerank("query", raw) == raw
    assert report.ui_reranker.rerank("query", raw) == raw
    assert reranker.rerank_calls == 1


def test_retriever_reranker_runtime_failure_returns_pre_rerank_results(tmp_path):
    frame_file = tmp_path / "frames" / "f1.jpg"
    frame_file.parent.mkdir()
    frame_file.write_bytes(b"frame")
    result = SearchResult("f1", "v1", 1.0, frame_path=str(frame_file))
    reranker = _Reranker(fail_rerank=True)

    class _Processor:
        def process(self, query, **kwargs):
            return SimpleNamespace(
                raw_query=query,
                visual_prompt=query,
                visual_prompt_vi=query,
                ocr_keywords=[],
                asr_keywords=[],
                metadata={},
            )

    class _Hydrator:
        def hydrate_with_diagnostics(self, results):
            return HydrationBatch([result], [])

    retriever = Retriever(
        experiment=_experiment(tmp_path),
        embedders={"model": _Embedder([], "model")},
        index=SimpleNamespace(),
        hydrator=_Hydrator(),
        query_processor=_Processor(),
        reranker=reranker,
    )
    retriever._load_frame_embeddings = lambda model: (
        __import__("numpy").asarray([[1.0]], dtype="float32"),
        [{"frame_id": "f1"}],
    )

    assert retriever.search("query", top_k=1) == [result]
    assert retriever.reranker is None
    assert reranker.rerank_calls == 1
