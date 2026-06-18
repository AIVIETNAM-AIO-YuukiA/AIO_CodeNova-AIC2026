"""Elasticsearch-backed text index.

Not yet wired into the pipeline — there is no OCR/ASR text to index yet. The
implementation is kept minimal and the ``elasticsearch`` package is imported
lazily so the project runs without it installed (install via the ``text`` extra).
"""

from __future__ import annotations

from codenova.core.errors import IndexBuildError
from codenova.stores.text.base import TextDocument, TextIndex


class ElasticTextIndex(TextIndex):
    """Full-text index stored in an Elasticsearch index."""

    def __init__(self, url: str, index_name: str, api_key: str | None = None) -> None:
        self.url = url
        self.index_name = index_name
        self.api_key = api_key
        self._client = None

    def index_documents(self, documents: list[TextDocument]) -> None:
        """Bulk-index text documents into Elasticsearch."""
        if not documents:
            return
        client = self._connect()
        operations = []
        for doc in documents:
            operations.append({"index": {"_index": self.index_name, "_id": doc.doc_id}})
            operations.append(
                {
                    "video_id": doc.video_id,
                    "frame_id": doc.frame_id,
                    "text": doc.text,
                    "source": doc.source,
                    "timestamp_sec": doc.timestamp_sec,
                }
            )
        client.bulk(operations=operations, refresh=True)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Run a BM25 match query and return ``(doc_id, score)`` pairs."""
        client = self._connect()
        response = client.search(
            index=self.index_name,
            query={"match": {"text": query}},
            size=top_k,
        )
        return [(hit["_id"], float(hit["_score"])) for hit in response["hits"]["hits"]]

    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise IndexBuildError(
                "Install the 'text' extra (elasticsearch) before using the text index."
            ) from exc
        self._client = Elasticsearch(self.url, api_key=self.api_key)
        return self._client
