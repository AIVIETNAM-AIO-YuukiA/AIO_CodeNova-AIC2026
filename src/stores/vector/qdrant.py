"""Qdrant-backed vector index.

Every point uses Qdrant named vectors, one name per embedding model (e.g.
"siglip2-large", "beit3-base"), even when there is only one — this keeps
build/search logic uniform regardless of how many models are configured.
Embeddings are L2-normalized upstream, so cosine distance ranks identically to
inner product. Each point stores its ``frame_id`` in the payload; the numeric
point id is just the row position, so searches return frame ids directly.
"""

from __future__ import annotations

from core.errors import IndexBuildError
from core.types import SearchResult
from stores.vector.base import VectorIndex, frame_result


class QdrantVectorIndex(VectorIndex):
    """Vector index stored in a Qdrant collection with named vectors."""

    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        distance: str = "Cosine",
        upsert_batch_size: int = 256,
    ) -> None:
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self.distance = distance
        self.upsert_batch_size = upsert_batch_size
        self._client = None

    def build(
        self, embeddings_by_model: dict[str, list[list[float]]], frame_ids: list[str]
    ) -> None:
        """Create the collection (one named vector per model) and upsert all points."""
        if not embeddings_by_model:
            raise IndexBuildError("Cannot build a Qdrant index with no embedding models.")
        for model_name, embeddings in embeddings_by_model.items():
            if not embeddings:
                raise IndexBuildError(f"Model '{model_name}' has zero embeddings.")
            if len(embeddings) != len(frame_ids):
                raise IndexBuildError(
                    f"Model '{model_name}' embedding count and frame_id count differ."
                )

        client, models = self._connect()
        if client.collection_exists(self.collection):
            client.delete_collection(self.collection)
        client.create_collection(
            collection_name=self.collection,
            vectors_config={
                model_name: models.VectorParams(
                    size=len(embeddings[0]),
                    distance=models.Distance[self.distance.upper()],
                )
                for model_name, embeddings in embeddings_by_model.items()
            },
        )

        num_points = len(frame_ids)
        for start in range(0, num_points, self.upsert_batch_size):
            stop = min(start + self.upsert_batch_size, num_points)
            points = [
                models.PointStruct(
                    id=index,
                    vector={
                        model_name: embeddings[index]
                        for model_name, embeddings in embeddings_by_model.items()
                    },
                    payload={"frame_id": frame_ids[index]},
                )
                for index in range(start, stop)
            ]
            client.upsert(collection_name=self.collection, points=points)

    def search(
        self, query_embedding: list[float], top_k: int, model_name: str | None = None
    ) -> list[SearchResult]:
        """Search the named vector for ``model_name`` and return frame ids with scores."""
        if model_name is None:
            raise IndexBuildError(
                "QdrantVectorIndex.search requires model_name (named vectors are always used)."
            )
        client, _ = self._connect()
        response = client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            using=model_name,
            limit=top_k,
        )
        results: list[SearchResult] = []
        for hit in response.points:
            frame_id = (hit.payload or {}).get("frame_id")
            if frame_id is None:
                continue
            results.append(frame_result(str(frame_id), float(hit.score)))
        return results

    def _connect(self):
        if self._client is not None:
            return self._client, self._models

        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise IndexBuildError("Install qdrant-client before using the Qdrant index.") from exc

        self._client = QdrantClient(url=self.url, api_key=self.api_key)
        self._models = models
        return self._client, self._models
