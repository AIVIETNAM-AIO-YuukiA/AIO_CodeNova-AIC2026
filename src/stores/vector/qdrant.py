"""Qdrant-backed vector index.

Every point uses Qdrant named vectors, one name per embedding model (e.g.
"siglip2-large", "beit3-base"), even when there is only one — this keeps
build/search logic uniform regardless of how many models are configured.
Embeddings are L2-normalized upstream, so cosine distance ranks identically to
inner product. Each point stores its ``frame_id`` in the payload; the numeric
point id is just the row position, so searches return frame ids directly.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from core.errors import IndexBuildError
from core.types import SearchResult
from stores.vector.base import VectorIndex, frame_result

if TYPE_CHECKING:
    import numpy as np


class QdrantVectorIndex(VectorIndex):
    """Vector index stored in a Qdrant collection with named vectors."""

    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        distance: str = "Cosine",
        upsert_batch_size: int = 256,
        upsert_workers: int = 4,
        upsert_max_retries: int = 3,
    ) -> None:
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self.distance = distance
        self.upsert_batch_size = upsert_batch_size
        self.upsert_workers = upsert_workers
        self.upsert_max_retries = upsert_max_retries
        self._client = None

    def build(self, embeddings_by_model: dict[str, "np.ndarray"], frame_ids: list[str]) -> None:
        """Create the collection (one named vector per model) and upsert all points."""
        if not embeddings_by_model:
            raise IndexBuildError("Cannot build a Qdrant index with no embedding models.")
        for model_name, embeddings in embeddings_by_model.items():
            if len(embeddings) == 0:
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
                    size=embeddings.shape[1],
                    distance=models.Distance[self.distance.upper()],
                )
                for model_name, embeddings in embeddings_by_model.items()
            },
        )

        num_points = len(frame_ids)
        batch_starts = range(0, num_points, self.upsert_batch_size)

        def upsert_batch(start: int) -> None:
            stop = min(start + self.upsert_batch_size, num_points)
            # .tolist() only on the slice being upserted, not the whole array,
            # so peak RAM stays close to the numpy arrays' own footprint.
            batch_by_model = {
                model_name: embeddings[start:stop].tolist()
                for model_name, embeddings in embeddings_by_model.items()
            }
            points = [
                models.PointStruct(
                    id=start + offset,
                    vector={
                        model_name: batch[offset] for model_name, batch in batch_by_model.items()
                    },
                    payload={"frame_id": frame_ids[start + offset]},
                )
                for offset in range(stop - start)
            ]
            # Each concurrent worker adds load on top of the others, so a
            # transient read timeout under that load is retried rather than
            # failing the whole multi-hour build over one slow batch.
            last_error: Exception | None = None
            for attempt in range(self.upsert_max_retries):
                try:
                    client.upsert(collection_name=self.collection, points=points)
                    return
                except Exception as exc:  # noqa: BLE001 - qdrant/httpx raise several types
                    last_error = exc
                    if attempt < self.upsert_max_retries - 1:
                        time.sleep(2.0 * (attempt + 1))
            raise IndexBuildError(
                f"Upsert failed for points [{start}, {stop}) after "
                f"{self.upsert_max_retries} attempts: {last_error}"
            ) from last_error

        # Upsert is I/O-bound (waiting on Qdrant per batch), so a thread pool
        # overlaps requests instead of waiting on each round-trip in turn.
        # Kept small (default 4) since a local Qdrant instance can time out
        # under too many concurrent large-payload upserts.
        with ThreadPoolExecutor(max_workers=self.upsert_workers) as pool:
            list(pool.map(upsert_batch, batch_starts))

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
            payload = hit.payload or {}
            frame_id = payload.get("frame_id")
            if frame_id is None:
                continue
            res = frame_result(str(frame_id), float(hit.score))
            if payload.get("frame_path"):
                import dataclasses

                res = dataclasses.replace(res, frame_path=payload["frame_path"])
            results.append(res)
        return results

    def get_vector(self, frame_id: str, model_name: str) -> list[float] | None:
        """Fetch one point's named vector via a payload filter, not a full scan."""
        client, models = self._connect()
        points, _ = client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="frame_id", match=models.MatchValue(value=frame_id))]
            ),
            with_vectors=[model_name],
            limit=1,
        )
        if not points:
            return None
        vector = points[0].vector
        return vector.get(model_name) if isinstance(vector, dict) else vector

    def _connect(self):
        if self._client is not None:
            return self._client, self._models

        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise IndexBuildError("Install qdrant-client before using the Qdrant index.") from exc

        # Default client timeout (5s) is too short once several upsert workers
        # are hitting the same instance concurrently with large payloads.
        self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60)
        self._models = models
        return self._client, self._models
