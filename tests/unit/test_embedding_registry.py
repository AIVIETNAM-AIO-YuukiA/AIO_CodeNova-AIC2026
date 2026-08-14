import pytest

import modules.embedding as embedding
from config.settings import Experiment, PipelineConfig
from core.errors import EmbeddingError
from core.errors import RetrievalError
from indexing.manifest import JsonlManifest
from modules.embedding import resolve_embedding_model, resolve_embedding_models
from retrieval.search import build_retriever


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        ("jina-clip-v2", "JinaClipEmbedder"),
        ("beit3-large", "Beit3Embedder"),
        ("siglip2-so400m", "SiglipEmbedder"),
        ("google/siglip2-so400m-patch14-384", "SiglipEmbedder"),
        ("vietnamese-embedding", "VietnameseEmbedder"),
        ("AITeamVN/Vietnamese_Embedding_v2", "VietnameseEmbedder"),
    ],
)
def test_registry_resolves_supported_aliases_and_model_ids(name, backend):
    spec = resolve_embedding_model(name)

    assert spec.requested_name == name
    assert spec.backend == backend
    assert spec.resolved_model_id
    assert spec.preprocessing


@pytest.mark.parametrize(
    "name",
    ["unknown-model", "siglp2-so400m", "siglip-typo", "beit3-typo", "", "clip-model"],
)
def test_registry_rejects_unknown_or_misspelled_models(name):
    with pytest.raises(EmbeddingError, match="Unsupported embedding model"):
        resolve_embedding_model(name)


def test_registry_rejects_duplicate_model_artifact_names():
    with pytest.raises(EmbeddingError, match="Duplicate"):
        resolve_embedding_models(("jina-clip-v2", "jina-clip-v2"))


def test_build_embedder_dispatches_explicit_siglip_marker(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(embedding, "SiglipEmbedder", lambda **kwargs: sentinel)

    assert embedding.build_embedder("google/siglip-custom", device="cpu") is sentinel


def test_jina_alias_uses_resolved_environment_model(monkeypatch):
    received = {}

    def fake_jina(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setenv("JINA_EMBEDDING_MODEL", "jinaai/custom-jina")
    monkeypatch.setattr(embedding, "JinaClipEmbedder", fake_jina)

    embedding.build_embedder("jina-clip-v2", device="cpu")

    assert resolve_embedding_model("jina-clip-v2").resolved_model_id == "jinaai/custom-jina"
    assert received["model_name"] is None


def test_build_embedder_rejects_unknown_before_constructing_backend(monkeypatch):
    monkeypatch.setattr(
        embedding,
        "SiglipEmbedder",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback constructed")),
    )

    with pytest.raises(EmbeddingError, match="unknown-model"):
        embedding.build_embedder("unknown-model", device="cpu")


def test_retrieval_rejects_runtime_model_resolution_different_from_offline(tmp_path, monkeypatch):
    experiment = Experiment(
        name="exp",
        run_dir=tmp_path,
        config=PipelineConfig(runs_dir=tmp_path, embedding_models=("jina-clip-v2",)),
    )
    JsonlManifest(tmp_path / "manifests" / "embeddings.jsonl").replace_all(
        [
            {
                "model_name": "jina-clip-v2",
                "requested_name": "jina-clip-v2",
                "backend": "JinaClipEmbedder",
                "resolved_model_id": "jinaai/jina-clip-v2",
                "revision": None,
                "preprocessing": "jina-clip-v2:image-512:l2-normalized",
            }
        ]
    )
    monkeypatch.setenv("JINA_EMBEDDING_MODEL", "jinaai/different-jina-model")

    with pytest.raises(RetrievalError, match="resolved_model_id"):
        build_retriever(experiment)
