"""Image/text embedding models (Jina CLIP v2, SigLIP 2, BEiT-3, Vietnamese captions)."""

from pathlib import Path

from modules.embedding.base import Embedder
from modules.embedding.beit3 import Beit3Embedder
from modules.embedding.jina import JinaClipEmbedder
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
    """Chon embedder theo ten trong ``model_name``.

    Ten chua "jina" -> Jina CLIP v2; "beit3" -> BEiT-3; nam trong
    _VIETNAMESE_MARKERS -> Vietnamese embedding (captions_path la noi cache
    caption VLM, chi backend nay dung); con lai -> SigLIP 2.

    Neu model_name la ten ngan (khong co "/", vd "vietnamese-embedding" hay
    "siglip2-so400m" dung de dispatch o tren) thi truyen None cho embedder -
    embedder se tu doc ten HuggingFace day du tu bien moi truong rieng cua no
    (VIETNAMESE_EMBEDDING_MODEL / SIGLIP2_EMBEDDING_MODEL trong .env).
    """
    lowered = model_name.lower()
    if "jina" in lowered:
        return JinaClipEmbedder(model_name=model_name, device=device, batch_size=batch_size)
    if "beit3" in lowered:
        return Beit3Embedder(model_name=model_name, device=device, batch_size=batch_size)
    if any(marker in lowered for marker in _VIETNAMESE_MARKERS):
        return VietnameseEmbedder(
            model_name=model_name if "/" in model_name else None,
            device=device,
            batch_size=batch_size,
            captions_path=captions_path,
        )
    return SiglipEmbedder(
        model_name=model_name if "/" in model_name else None,
        device=device,
        batch_size=batch_size,
    )


__all__ = [
    "Beit3Embedder",
    "Embedder",
    "JinaClipEmbedder",
    "SiglipEmbedder",
    "VietnameseEmbedder",
    "build_embedder",
]
