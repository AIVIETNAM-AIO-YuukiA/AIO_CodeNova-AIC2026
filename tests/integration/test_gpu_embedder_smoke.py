"""Explicitly enabled smoke test for one real GPU embedder.

The model name must be supplied so CI never downloads or loads a large model
implicitly::

    RUN_GPU_TESTS=1 CODENOVA_GPU_SMOKE_MODEL=jina-clip-v2 \
      pytest -q tests/integration/test_gpu_embedder_smoke.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from modules.embedding import build_embedder


pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]


def _gpu_test_configuration() -> tuple[bool, str | None]:
    return os.getenv("RUN_GPU_TESTS") == "1", os.getenv("CODENOVA_GPU_SMOKE_MODEL")


@pytest.mark.skipif(
    not all(_gpu_test_configuration()),
    reason="set RUN_GPU_TESTS=1 and CODENOVA_GPU_SMOKE_MODEL",
)
def test_real_embedder_produces_finite_cuda_query_vector() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _, model_name = _gpu_test_configuration()
    assert model_name is not None

    embedder = build_embedder(model_name=model_name, device="cuda")
    vector = np.asarray(embedder.embed_text("offline indexing smoke test"))

    assert vector.size > 0
    assert np.isfinite(vector).all()
