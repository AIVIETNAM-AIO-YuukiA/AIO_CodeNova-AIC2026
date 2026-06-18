import pytest

from codenova.config.settings import Experiment, PipelineConfig
from codenova.stores.text import TextDocument, TextIndex, build_text_index
from codenova.stores.text.elasticsearch import ElasticTextIndex
from codenova.stores.vector import QdrantVectorIndex, VectorIndex
from codenova.stores.vector.factory import build_vector_index


def test_build_vector_index_returns_qdrant_with_namespaced_collection(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://example:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "frames")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    experiment = Experiment(name="exp1", run_dir=tmp_path, config=PipelineConfig(runs_dir=tmp_path))

    index = build_vector_index(experiment)

    assert isinstance(index, QdrantVectorIndex)
    assert isinstance(index, VectorIndex)
    assert index.url == "http://example:6333"
    assert index.collection == "frames__exp1"


def test_build_vector_index_rejects_unknown_backend(tmp_path) -> None:
    config = PipelineConfig(runs_dir=tmp_path, index_backend="milvus")
    experiment = Experiment(name="exp1", run_dir=tmp_path, config=config)

    with pytest.raises(Exception, match="Unsupported index backend"):
        build_vector_index(experiment)


def test_text_index_interface_is_abstract() -> None:
    doc = TextDocument(doc_id="d1", video_id="v1", text="hello", source="ocr")
    assert doc.source == "ocr"
    with pytest.raises(NotImplementedError):
        TextIndex().search("q", top_k=5)


def test_build_text_index_reads_env_and_namespaces_index(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ELASTIC_URL", "http://example:9200")
    monkeypatch.setenv("ELASTIC_INDEX", "text")
    monkeypatch.delenv("ELASTIC_API_KEY", raising=False)
    experiment = Experiment(name="exp1", run_dir=tmp_path, config=PipelineConfig(runs_dir=tmp_path))

    index = build_text_index(experiment)

    assert isinstance(index, ElasticTextIndex)
    assert isinstance(index, TextIndex)
    assert index.url == "http://example:9200"
    assert index.index_name == "text__exp1"
