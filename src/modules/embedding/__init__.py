"""Image/text embedding models (SigLIP 2, BEiT-3, Vietnamese caption embedding)."""

from pathlib import Path

from modules.embedding.base import Embedder
from modules.embedding.beit3 import Beit3Embedder
from modules.embedding.siglip import SiglipEmbedder
from modules.embedding.vietnamese import VietnameseEmbedder

# Model names that route to the Vietnamese caption-embedding backend rather
# than a direct image embedder. Matched by substring, same convention as the
# "beit3" check below.
_VIETNAMESE_MARKERS = ("vietnamese-embedding", "vietnamese_embedding")


def build_embedder(
    model_name: str,
    device: str = "auto",
    batch_size: int = 32,
    captions_path: Path | None = None,
) -> Embedder:
    """Return the embedder matching ``model_name``.

    Names containing "beit3" use the BEiT-3 backend; names containing
    "vietnamese-embedding" use the caption-then-embed Vietnamese backend
    (``captions_path`` is where it caches VLM-generated captions — required
    for that backend, ignored by the others); everything else (SigLIP 2
    checkpoints) uses the SigLIP backend.
    """
    lowered = model_name.lower()
    if "beit3" in lowered:
        return Beit3Embedder(model_name=model_name, device=device, batch_size=batch_size)
    if any(marker in lowered for marker in _VIETNAMESE_MARKERS):
        return VietnameseEmbedder(
            model_name=model_name if "/" in model_name else None,
            device=device,
            batch_size=batch_size,
            captions_path=captions_path,
        )
    return SiglipEmbedder(model_name=model_name, device=device, batch_size=batch_size)


__all__ = [
    "Beit3Embedder",
    "Embedder",
    "SiglipEmbedder",
    "VietnameseEmbedder",
    "build_embedder",
]
