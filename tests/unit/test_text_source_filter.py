import pytest

from core.errors import IndexBuildError
from stores.text.base import TextDocument
from stores.text.elasticsearch import ElasticTextIndex


class FakeElasticClient:
    def __init__(self) -> None:
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"hits": {"hits": []}}


def test_elasticsearch_uses_terms_filter_for_multiple_text_sources():
    index = ElasticTextIndex("http://example", "text")
    client = FakeElasticClient()
    index._client = client

    assert index.search_documents("query", 10, source=("ocr", "asr")) == []

    source_filter = client.calls[0]["query"]["bool"]["filter"]
    assert source_filter == [{"terms": {"source": ["ocr", "asr"]}}]


def test_elasticsearch_skips_request_for_empty_source_list():
    index = ElasticTextIndex("http://example", "text")
    client = FakeElasticClient()
    index._client = client

    assert index.search_documents("query", 10, source=[]) == []
    assert client.calls == []


def test_elasticsearch_indexes_large_import_in_bounded_batches(monkeypatch):
    class FakeIndices:
        def exists(self, **kwargs):
            return True

    class FakeBulkClient:
        def __init__(self):
            self.indices = FakeIndices()
            self.bulk_calls = []

        def bulk(self, **kwargs):
            self.bulk_calls.append(kwargs)
            return {"errors": False}

    client = FakeBulkClient()
    index = ElasticTextIndex("http://example", "text")
    index._client = client
    monkeypatch.setattr("stores.text.elasticsearch.BULK_INDEX_BATCH_SIZE", 2)
    documents = [TextDocument(f"d{i}", "v1", "sample text", "asr") for i in range(5)]

    index.index_documents(documents)

    assert [len(call["operations"]) for call in client.bulk_calls] == [4, 4, 2]
    assert [call["refresh"] for call in client.bulk_calls] == [False, False, True]


def test_elasticsearch_raises_when_a_bulk_batch_has_document_errors():
    class FakeIndices:
        def exists(self, **kwargs):
            return True

    class FakeBulkClient:
        indices = FakeIndices()

        def bulk(self, **kwargs):
            return {"errors": True}

    index = ElasticTextIndex("http://example", "text")
    index._client = FakeBulkClient()

    with pytest.raises(IndexBuildError, match="rejected"):
        index.index_documents([TextDocument("d1", "v1", "sample", "ocr")])
