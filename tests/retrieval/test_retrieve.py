"""Tests for connoisseur.retrieval.retrieve — similarity search wrappers."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import connoisseur.retrieval.retrieve as retrieve_module
from connoisseur.retrieval.retrieve import (
    retrieve_articles,
    retrieve_images_by_image,
    print_hits,
    _unwrap,
)


# ---------------------------------------------------------------------------
# Fixture: reset cached DB singletons
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db_singletons():
    original_article = retrieve_module._article_db
    original_image = retrieve_module._image_db
    retrieve_module._article_db = None
    retrieve_module._image_db = None
    yield
    retrieve_module._article_db = original_article
    retrieve_module._image_db = original_image


# ---------------------------------------------------------------------------
# _unwrap
# ---------------------------------------------------------------------------

def test_unwrap_extracts_first_element():
    res = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"a": 1}, {"b": 2}]],
        "distances": [[0.1, 0.2]],
    }
    ids, docs, metas, dists = _unwrap(res)
    assert ids == ["id1", "id2"]
    assert docs == ["doc1", "doc2"]
    assert metas == [{"a": 1}, {"b": 2}]
    assert dists == [0.1, 0.2]


def test_unwrap_handles_empty():
    res = {}
    ids, docs, metas, dists = _unwrap(res)
    assert ids == []
    assert docs == []
    assert metas == []
    assert dists == []


# ---------------------------------------------------------------------------
# retrieve_articles
# ---------------------------------------------------------------------------

@patch("connoisseur.retrieval.retrieve.embed_texts")
@patch("connoisseur.retrieval.retrieve.Chroma")
def test_retrieve_articles_returns_four_tuple(MockChroma, mock_embed):
    mock_embed.return_value = np.ones((1, 384), dtype=np.float32)
    col = MagicMock()
    col.query.return_value = {
        "ids": [["r1"]],
        "documents": [["text"]],
        "metadatas": [[{"doc_id": "r1"}]],
        "distances": [[0.2]],
    }
    MockChroma.return_value._collection = col

    ids, docs, metas, dists = retrieve_articles("Italian food", k=1)

    assert ids == ["r1"]
    assert docs == ["text"]
    assert len(metas) == 1
    assert len(dists) == 1


@patch("connoisseur.retrieval.retrieve.embed_texts")
@patch("connoisseur.retrieval.retrieve.Chroma")
def test_retrieve_articles_passes_k_to_query(MockChroma, mock_embed):
    mock_embed.return_value = np.ones((1, 384), dtype=np.float32)
    col = MagicMock()
    col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    MockChroma.return_value._collection = col

    retrieve_articles("query", k=7)

    call_kwargs = col.query.call_args[1]
    assert call_kwargs["n_results"] == 7


@patch("connoisseur.retrieval.retrieve.embed_texts")
@patch("connoisseur.retrieval.retrieve.Chroma")
def test_retrieve_articles_passes_where_filter(MockChroma, mock_embed):
    mock_embed.return_value = np.ones((1, 384), dtype=np.float32)
    col = MagicMock()
    col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    MockChroma.return_value._collection = col

    where = {"cuisine": "Italian"}
    retrieve_articles("pasta", where=where)

    call_kwargs = col.query.call_args[1]
    assert call_kwargs["where"] == where


@patch("connoisseur.retrieval.retrieve.embed_texts")
@patch("connoisseur.retrieval.retrieve.Chroma")
def test_retrieve_articles_singleton_db(MockChroma, mock_embed):
    mock_embed.return_value = np.ones((1, 384), dtype=np.float32)
    col = MagicMock()
    col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    MockChroma.return_value._collection = col

    retrieve_articles("a")
    retrieve_articles("b")

    MockChroma.assert_called_once()


# ---------------------------------------------------------------------------
# retrieve_images_by_image
# ---------------------------------------------------------------------------

@patch("connoisseur.retrieval.retrieve.embed_images")
@patch("connoisseur.retrieval.retrieve.Chroma")
def test_retrieve_images_by_image_returns_four_tuple(MockChroma, mock_embed):
    mock_embed.return_value = np.ones((1, 512), dtype=np.float32)
    col = MagicMock()
    col.query.return_value = {
        "ids": [["img_0"]],
        "documents": [["recipe name"]],
        "metadatas": [[{"doc_id": "img_0"}]],
        "distances": [[0.1]],
    }
    MockChroma.return_value._collection = col

    ids, docs, metas, dists = retrieve_images_by_image("path/to/query.png")

    assert ids == ["img_0"]
    assert docs == ["recipe name"]


# ---------------------------------------------------------------------------
# print_hits
# ---------------------------------------------------------------------------

def test_print_hits_output(capsys):
    print_hits(
        ids=["r1"],
        docs=["Great Italian restaurant"],
        metas=[{"doc_id": "r1", "cuisine": "Italian", "location": "LA", "source": "restaurant"}],
        dists=[0.15],
        title="Search Results",
    )
    captured = capsys.readouterr()
    assert "Search Results" in captured.out
    assert "Italian" in captured.out
    assert "LA" in captured.out


def test_print_hits_truncates_long_snippet(capsys):
    long_doc = "X" * 300
    print_hits(["r1"], [long_doc], [{}], [0.5], "Truncation Test", max_chars=50)
    captured = capsys.readouterr()
    assert "..." in captured.out


def test_print_hits_empty(capsys):
    print_hits([], [], [], [], "Empty")
    captured = capsys.readouterr()
    assert "Empty" in captured.out


def test_print_hits_handles_missing_meta(capsys):
    print_hits(["r1"], ["doc"], [None], [0.2], "No Meta")
    captured = capsys.readouterr()
    assert "N/A" in captured.out
