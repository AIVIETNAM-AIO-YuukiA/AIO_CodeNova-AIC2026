"""Elasticsearch-backed text index.

Indexes OCR (``modules/ocr/vllm.py``) and ASR (``modules/asr/gipformer.py``)
documents under one mapping (same convention as the AIC 2025 reference
project's ``elasticsearch_service.py``: one index, ``source`` field
distinguishes OCR vs ASR rather than separate indices per modality).
"""

from __future__ import annotations

from core.errors import IndexBuildError
from stores.text.base import TextDocument, TextIndex

# Custom analyzer for Vietnamese/multilingual text: lowercases, strips
# diacritics-as-separate-chars (asciifolding), and drops English stopwords —
# same setup as the AIC 2025 reference project. ``text.exact`` is a
# keyword-tokenized sub-field for substring/exact-phrase matching, which the
# standard analyzer's tokenization can't do.
_INDEX_MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "multilingual_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "stop"],
                },
                "exact_analyzer": {
                    "type": "custom",
                    "tokenizer": "keyword",
                    "filter": ["lowercase"],
                },
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "multilingual_analyzer",
                "fields": {"exact": {"type": "text", "analyzer": "exact_analyzer"}},
            },
            "video_id": {"type": "keyword"},
            "frame_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "timestamp_sec": {"type": "float"},
        }
    },
}


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
        self._ensure_index(client)
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
        """Run a multi-strategy BM25 query and return ``(doc_id, score)`` pairs.

        Combines exact-phrase (highest weight), all-words, most-words, and
        fuzzy matching in one query so typos and partial matches still rank
        below an exact hit rather than needing separate calls.
        """
        client = self._connect()
        should = [
            {"match_phrase": {"text": {"query": query, "boost": 8.0}}},
            {"match": {"text": {"query": query, "operator": "and", "boost": 5.0}}},
            {
                "match": {
                    "text": {
                        "query": query,
                        "operator": "or",
                        "minimum_should_match": "60%",
                        "boost": 3.0,
                    }
                }
            },
            {"match": {"text": {"query": query, "fuzziness": "AUTO", "boost": 2.0}}},
        ]
        response = client.search(
            index=self.index_name,
            query={"bool": {"should": should, "minimum_should_match": 1}},
            size=top_k,
        )
        return [(hit["_id"], float(hit["_score"])) for hit in response["hits"]["hits"]]

    def export_all(self):
        """Yield every document in the index as a dict, for local backup/inspection.

        Uses the scroll API since ``search`` is capped at ``top_k`` — indices
        here can hold hundreds of thousands of OCR/ASR documents.
        """
        from elasticsearch.helpers import scan

        client = self._connect()
        for hit in scan(client, index=self.index_name, query={"query": {"match_all": {}}}):
            yield {"doc_id": hit["_id"], **hit["_source"]}

    def _ensure_index(self, client) -> None:
        """Create the index with the custom analyzer mapping if it doesn't exist yet.

        Without this, Elasticsearch's dynamic mapping would guess a plain
        ``text`` field on first write and the ``multilingual_analyzer``/
        ``text.exact`` sub-field would never apply.
        """
        if client.indices.exists(index=self.index_name):
            return
        client.indices.create(index=self.index_name, body=_INDEX_MAPPING)

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
