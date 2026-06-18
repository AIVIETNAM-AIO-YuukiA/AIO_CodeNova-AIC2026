import pytest

from retrieval.tracks import TrackQuery, build_retrieval_text


def test_textual_kis_uses_query_first() -> None:
    request = TrackQuery(
        track="textual_kis",
        query="moving motorbike",
        context="under the rider left arm",
    )

    assert build_retrieval_text(request) == "moving motorbike under the rider left arm"


def test_vqa_combines_context_question_and_query() -> None:
    request = TrackQuery(
        track="vqa",
        query="telephone conversation",
        context="two women talk and one hangs pictures",
        question="What are the names of the two women?",
    )

    assert build_retrieval_text(request) == (
        "two women talk and one hangs pictures "
        "What are the names of the two women? telephone conversation"
    )


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_retrieval_text(TrackQuery(track="textual_kis"))
