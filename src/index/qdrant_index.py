"""Qdrant-backed vector index.

Embeddings are L2-normalized upstream, so cosine distance ranks identically to
inner product. Each point stores its ``frame_id`` in the payload; the numeric
point id is just the row position, so searches return frame ids directly.
"""

from __future__ import annotations

from core.errors import IndexBuildError
from index.base import VectorIndex, frame_result
from core.types import SearchResult


class QdrantVectorIndex(VectorIndex):
    """Vector index stored in a Qdrant collection."""

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

    def build(self, embeddings: list[list[float]], frame_ids: list[str]) -> None:
        """Create the collection and upsert all embeddings."""
        if not embeddings:
            raise IndexBuildError("Cannot build a Qdrant index with zero embeddings.")
        if len(embeddings) != len(frame_ids):
            raise IndexBuildError("Embedding count and frame_id count differ.")

        client, models = self._connect()
        if client.collection_exists(self.collection):
            client.delete_collection(self.collection)
        client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=len(embeddings[0]),
                distance=models.Distance[self.distance.upper()],
            ),
        )
        for start in range(0, len(embeddings), self.upsert_batch_size):
            stop = start + self.upsert_batch_size
            points = [
                models.PointStruct(
                    id=index,
                    vector=embeddings[index],
                    payload={"frame_id": frame_ids[index]},
                )
                for index in range(start, min(stop, len(embeddings)))
            ]
            client.upsert(collection_name=self.collection, points=points)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Search the Qdrant collection and return frame ids with scores."""
        client, _ = self._connect()
        response = client.query_points(
            collection_name=self.collection,
            query=query_embedding,
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
