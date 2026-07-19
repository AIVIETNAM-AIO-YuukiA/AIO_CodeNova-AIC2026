from types import SimpleNamespace

from modules.embedding.base import projected_features


def test_projected_features_accepts_plain_tensor() -> None:
    tensor = object()

    assert projected_features(tensor) is tensor


def test_projected_features_accepts_transformers_output() -> None:
    tensor = object()
    output = SimpleNamespace(pooler_output=tensor)

    assert projected_features(output) is tensor
