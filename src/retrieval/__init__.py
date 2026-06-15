"""Text-to-video retrieval."""

from retrieval.query_processor import (
    ProcessedQuery,
    QueryProcessor,
    PassThroughQueryProcessor,
    LlmQueryProcessor,
    get_query_processor,
)
