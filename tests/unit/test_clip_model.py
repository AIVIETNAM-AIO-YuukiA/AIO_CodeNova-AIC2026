from types import SimpleNamespace

from embedding.clip_model import clip_features_tensor


def test_clip_features_tensor_accepts_plain_tensor() -> None:
    tensor = object()

    assert clip_features_tensor(tensor) is tensor


def test_clip_features_tensor_accepts_transformers_output() -> None:
    tensor = object()
    output = SimpleNamespace(pooler_output=tensor)

    assert clip_features_tensor(output) is tensor
