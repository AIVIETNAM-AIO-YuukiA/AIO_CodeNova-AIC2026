"""Strict embedding backend resolution and artifact provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os

from core.errors import EmbeddingError


@dataclass(frozen=True)
class EmbeddingModelSpec:
    requested_name: str
    backend: str
    resolved_model_id: str
    revision: str | None
    preprocessing: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


_ALIASES = {
    "jina": "jina",
    "jina-clip-v2": "jina",
    "beit3": "beit3",
    "beit3-large": "beit3",
    "siglip2": "siglip",
    "siglip2-so400m": "siglip",
    "siglip2-large": "siglip",
    "vietnamese-embedding": "vietnamese",
    "vietnamese_embedding": "vietnamese",
}

_BACKEND_NAMES = {
    "jina": "JinaClipEmbedder",
    "beit3": "Beit3Embedder",
    "siglip": "SiglipEmbedder",
    "vietnamese": "VietnameseEmbedder",
}

_PREPROCESSING = {
    "jina": "jina-clip-v2:image-512:l2-normalized",
    "beit3": "beit3:patch16-image-384:imagenet-normalized:l2-normalized",
    "siglip": "siglip2:image-384:max-text-64:l2-normalized",
    "vietnamese": "vlm-caption:vietnamese-sentence-embedding:l2-normalized",
}


def resolve_embedding_model(model_name: str) -> EmbeddingModelSpec:
    """Resolve only an explicit alias or a model ID with a known marker."""
    requested = model_name.strip()
    lowered = requested.lower()
    family = _ALIASES.get(lowered)
    if family is None and "/" in requested:
        if "jina" in lowered:
            family = "jina"
        elif "vietnamese-embedding" in lowered or "vietnamese_embedding" in lowered:
            family = "vietnamese"
        elif "siglip" in lowered:
            family = "siglip"
    if family is None:
        aliases = ", ".join(sorted(_ALIASES))
        raise EmbeddingError(
            f"Unsupported embedding model {model_name!r}. Supported aliases: {aliases}; "
            "custom repository IDs must contain jina, siglip, or vietnamese-embedding."
        )

    if family == "jina":
        resolved = (
            requested
            if "/" in requested
            else os.getenv("JINA_EMBEDDING_MODEL", "jinaai/jina-clip-v2")
        )
        revision = None
    elif family == "siglip":
        resolved = (
            requested
            if "/" in requested
            else os.getenv("SIGLIP2_EMBEDDING_MODEL", "google/siglip2-so400m-patch14-384")
        )
        revision = None
    elif family == "vietnamese":
        resolved = (
            requested
            if "/" in requested
            else os.getenv("VIETNAMESE_EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding_v2")
        )
        revision = None
    else:
        resolved = requested
        revision = None

    return EmbeddingModelSpec(
        requested_name=requested,
        backend=_BACKEND_NAMES[family],
        resolved_model_id=resolved,
        revision=revision,
        preprocessing=_PREPROCESSING[family],
    )


def resolve_embedding_models(model_names: tuple[str, ...]) -> tuple[EmbeddingModelSpec, ...]:
    if not model_names:
        raise EmbeddingError("At least one embedding model must be configured.")
    specs = tuple(resolve_embedding_model(name) for name in model_names)
    requested = [spec.requested_name for spec in specs]
    if len(requested) != len(set(requested)):
        raise EmbeddingError(f"Duplicate embedding model names are not allowed: {requested}")
    return specs
