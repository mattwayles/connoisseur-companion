"""Tests for connoisseur.retrieval.fusion — score utilities and cross-modal fusion."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from connoisseur.retrieval.fusion import (
    _to_similarity,
    _minmax,
    fuse_rank,
    print_fused,
)

# ---------------------------------------------------------------------------
# _to_similarity
# ---------------------------------------------------------------------------

def test_to_similarity_converts_zero_distance():
    result = _to_similarity([0.0])
    np.testing.assert_allclose(result, [1.0])


def test_to_similarity_converts_one_distance():
    result = _to_similarity([1.0])
    np.testing.assert_allclose(result, [0.0])


def test_to_similarity_converts_multiple():
    result = _to_similarity([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_to_similarity_returns_float32():
    result = _to_similarity([0.5])
    assert result.dtype == np.float32


def test_to_similarity_accepts_numpy_array():
    arr = np.array([0.2, 0.8], dtype=np.float64)
    result = _to_similarity(arr)
    np.testing.assert_allclose(result, [0.8, 0.2], atol=1e-6)


# ---------------------------------------------------------------------------
# _minmax
# ---------------------------------------------------------------------------

def test_minmax_normalises_range():
    result = _minmax([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


def test_minmax_all_same_values_returns_ones():
    result = _minmax([5.0, 5.0, 5.0])
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0])


def test_minmax_single_element():
    result = _minmax([7.0])
    np.testing.assert_allclose(result, [1.0])


def test_minmax_empty_array():
    result = _minmax([])
    assert len(result) == 0


def test_minmax_returns_float32():
    result = _minmax([1.0, 2.0])
    assert result.dtype == np.float32


def test_minmax_output_bounded_zero_one():
    import random
    values = [random.uniform(-10, 10) for _ in range(20)]
    result = _minmax(values)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


# ---------------------------------------------------------------------------
# fuse_rank
# ---------------------------------------------------------------------------

def _make_chroma_result(ids, docs, metas, distances):
    return {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


@patch("connoisseur.retrieval.fusion._get_image_db")
@patch("connoisseur.retrieval.fusion._get_article_db")
@patch("connoisseur.retrieval.fusion.embed_query_clip_text")
@patch("connoisseur.retrieval.fusion.embed_texts")
def test_fuse_rank_returns_list(mock_text_emb, mock_clip_emb, mock_art_db, mock_img_db):
    mock_text_emb.return_value = np.ones(384, dtype=np.float32)
    mock_clip_emb.return_value = np.ones(512, dtype=np.float32)

    art_col = MagicMock()
    art_col.query.return_value = _make_chroma_result(
        ["rest_0"], ["Restaurant A text"], [{"doc_id": "rest_0", "cuisine": "Italian", "location": "LA", "source": "restaurant"}], [0.2]
    )
    mock_art_db.return_value._collection = art_col

    img_col = MagicMock()
    img_col.query.return_value = _make_chroma_result(
        ["img_0"], ["Pasta"], [{"doc_id": "img_0", "cuisine": "Italian", "location": "N/A", "source": "recipe_image"}], [0.3]
    )
    mock_img_db.return_value._collection = img_col

    results = fuse_rank("Italian food", k_text=1, k_img=1)

    assert isinstance(results, list)
    assert len(results) == 2


@patch("connoisseur.retrieval.fusion._get_image_db")
@patch("connoisseur.retrieval.fusion._get_article_db")
@patch("connoisseur.retrieval.fusion.embed_query_clip_text")
@patch("connoisseur.retrieval.fusion.embed_texts")
def test_fuse_rank_sorted_by_fused_score(mock_text_emb, mock_clip_emb, mock_art_db, mock_img_db):
    mock_text_emb.return_value = np.ones(384, dtype=np.float32)
    mock_clip_emb.return_value = np.ones(512, dtype=np.float32)

    art_col = MagicMock()
    art_col.query.return_value = _make_chroma_result(
        ["r1", "r2"],
        ["Low relevance", "High relevance"],
        [{"doc_id": "r1", "cuisine": "X", "location": "Y", "source": "s"},
         {"doc_id": "r2", "cuisine": "X", "location": "Y", "source": "s"}],
        [0.9, 0.1],  # r1 farther, r2 closer
    )
    mock_art_db.return_value._collection = art_col

    img_col = MagicMock()
    img_col.query.return_value = _make_chroma_result([], [], [], [])
    mock_img_db.return_value._collection = img_col

    results = fuse_rank("query", k_text=2, k_img=0, top_n=2)

    # Higher fused score should come first (lower distance → higher similarity)
    assert results[0]["fused"] >= results[1]["fused"]


@patch("connoisseur.retrieval.fusion._get_image_db")
@patch("connoisseur.retrieval.fusion._get_article_db")
@patch("connoisseur.retrieval.fusion.embed_query_clip_text")
@patch("connoisseur.retrieval.fusion.embed_texts")
def test_fuse_rank_respects_top_n(mock_text_emb, mock_clip_emb, mock_art_db, mock_img_db):
    mock_text_emb.return_value = np.ones(384, dtype=np.float32)
    mock_clip_emb.return_value = np.ones(512, dtype=np.float32)

    art_col = MagicMock()
    art_col.query.return_value = _make_chroma_result(
        ["r1", "r2", "r3"],
        ["A", "B", "C"],
        [{"doc_id": f"r{i}", "cuisine": "X", "location": "Y", "source": "s"} for i in range(3)],
        [0.1, 0.2, 0.3],
    )
    mock_art_db.return_value._collection = art_col

    img_col = MagicMock()
    img_col.query.return_value = _make_chroma_result([], [], [], [])
    mock_img_db.return_value._collection = img_col

    results = fuse_rank("query", k_text=3, k_img=0, top_n=2)

    assert len(results) == 2


@patch("connoisseur.retrieval.fusion._get_image_db")
@patch("connoisseur.retrieval.fusion._get_article_db")
@patch("connoisseur.retrieval.fusion.embed_query_clip_text")
@patch("connoisseur.retrieval.fusion.embed_texts")
def test_fuse_rank_row_has_required_keys(mock_text_emb, mock_clip_emb, mock_art_db, mock_img_db):
    mock_text_emb.return_value = np.ones(384, dtype=np.float32)
    mock_clip_emb.return_value = np.ones(512, dtype=np.float32)

    art_col = MagicMock()
    art_col.query.return_value = _make_chroma_result(
        ["r1"], ["text"], [{"doc_id": "r1", "cuisine": "X", "location": "Y", "source": "s"}], [0.2]
    )
    mock_art_db.return_value._collection = art_col

    img_col = MagicMock()
    img_col.query.return_value = _make_chroma_result([], [], [], [])
    mock_img_db.return_value._collection = img_col

    results = fuse_rank("query", k_img=0)

    required_keys = {"modality", "id", "cuisine", "location", "source", "text_score", "img_score", "fused", "snippet"}
    assert required_keys.issubset(set(results[0].keys()))


# ---------------------------------------------------------------------------
# print_fused
# ---------------------------------------------------------------------------

def test_print_fused_outputs_rows(capsys):
    rows = [
        {
            "modality": "article", "id": "r1", "cuisine": "Italian", "location": "LA",
            "source": "restaurant", "text_score": 0.9, "img_score": 0.0, "fused": 0.9,
            "snippet": "Great Italian food",
        },
    ]
    print_fused(rows, "Test Results")
    captured = capsys.readouterr()
    assert "Test Results" in captured.out
    assert "Italian" in captured.out
    assert "r1" in captured.out


def test_print_fused_truncates_long_snippets(capsys):
    rows = [
        {
            "modality": "article", "id": "r1", "cuisine": "X", "location": "Y",
            "source": "s", "text_score": 1.0, "img_score": 0.0, "fused": 1.0,
            "snippet": "A" * 200,  # longer than default max_chars=90
        }
    ]
    print_fused(rows, "Truncation Test", max_chars=90)
    captured = capsys.readouterr()
    # Should be truncated with ...
    assert "..." in captured.out


def test_print_fused_empty_rows(capsys):
    print_fused([], "Empty")
    captured = capsys.readouterr()
    assert "Empty" in captured.out
