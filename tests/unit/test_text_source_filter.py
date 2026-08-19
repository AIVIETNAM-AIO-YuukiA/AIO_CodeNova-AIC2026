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
