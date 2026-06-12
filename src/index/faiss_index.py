"""FAISS vector index interface."""

from __future__ import annotations

from pathlib import Path
import json

from core.errors import IndexBuildError
from core.types import SearchResult


class VectorIndex:
    """Interface for vector index implementations."""

    def build(self, embeddings: list[list[float]], frame_ids: list[str]) -> None:
        """Build an index from embeddings."""
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Search the index."""
        raise NotImplementedError


class FaissVectorIndex(VectorIndex):
    """FAISS-backed inner-product index with optional GPU acceleration."""

    def __init__(
        self,
        index_path: Path,
        mapping_path: Path,
        use_gpu: bool = True,
        require_gpu: bool = True,
    ) -> None:
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.use_gpu = use_gpu
        self.require_gpu = require_gpu
        self._index = None
        self._frame_ids: list[str] = []

    def build(self, embeddings: list[list[float]], frame_ids: list[str]) -> None:
        """Build and persist a FAISS index."""
        if not embeddings:
            raise IndexBuildError("Cannot build a FAISS index with zero embeddings.")
        if len(embeddings) != len(frame_ids):
            raise IndexBuildError("Embedding count and frame_id count differ.")

        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise IndexBuildError("Install faiss-gpu-cu12 or faiss-cpu before indexing.") from exc

        vectors = np.asarray(embeddings, dtype="float32")
        cpu_index = faiss.IndexFlatIP(vectors.shape[1])
        index = self._maybe_to_gpu(faiss, cpu_index)
        index.add(vectors)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._to_cpu(faiss, index), str(self.index_path))
        self.mapping_path.write_text(json.dumps(frame_ids, indent=2) + "\n", encoding="utf-8")
        self._index = index
        self._frame_ids = frame_ids

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Search the FAISS index and return frame IDs with scores."""
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise IndexBuildError("Install faiss-gpu-cu12 or faiss-cpu before searching.") from exc

        if self._index is None:
            if not self.index_path.exists() or not self.mapping_path.exists():
                raise IndexBuildError("FAISS index or mapping file is missing.")
            cpu_index = faiss.read_index(str(self.index_path))
            self._index = self._maybe_to_gpu(faiss, cpu_index)
            self._frame_ids = json.loads(self.mapping_path.read_text(encoding="utf-8"))

        query = np.asarray([query_embedding], dtype="float32")
        scores, indices = self._index.search(query, top_k)
        results: list[SearchResult] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            results.append(frame_result(self._frame_ids[int(index)], float(score)))
        return results

    def _maybe_to_gpu(self, faiss, index):
        if not self.use_gpu:
            return index
        if hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
            resources = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(resources, 0, index)
        if self.require_gpu:
            raise IndexBuildError("FAISS GPU APIs are unavailable in the installed faiss package.")
        return index

    @staticmethod
    def _to_cpu(faiss, index):
        if hasattr(faiss, "index_gpu_to_cpu"):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception:
                return index
        return index


def frame_result(frame_id: str, score: float) -> SearchResult:
    """Create a minimal FAISS search result."""
    return SearchResult(frame_id=frame_id, video_id="", score=score)
