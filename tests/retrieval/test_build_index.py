"""Tests for connoisseur.retrieval.build_index — ChromaDB construction."""
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from connoisseur.retrieval.build_index import _build_article_docs, _build_image_docs, build_index


# ---------------------------------------------------------------------------
# _build_article_docs
# ---------------------------------------------------------------------------

def test_build_article_docs_creates_document(sample_restaurant):
    docs = _build_article_docs([sample_restaurant])
    assert len(docs) == 1
    assert "Mar de Cortez" in docs[0].page_content


def test_build_article_docs_skips_empty_name():
    restaurants = [
        {"name": "", "food_style": "Italian", "location": "LA"},
        {"name": "Valid Place", "food_style": "French", "location": "SF"},
    ]
    docs = _build_article_docs(restaurants)
    assert len(docs) == 1
    assert "Valid Place" in docs[0].page_content


def test_build_article_docs_skips_missing_name():
    restaurants = [{"food_style": "Thai", "location": "NYC"}]
    docs = _build_article_docs(restaurants)
    assert len(docs) == 0


def test_build_article_docs_metadata_has_doc_id(sample_restaurant):
    docs = _build_article_docs([sample_restaurant])
    assert "doc_id" in docs[0].metadata
    assert docs[0].metadata["doc_id"].startswith("rest_")


def test_build_article_docs_metadata_has_cuisine(sample_restaurant):
    docs = _build_article_docs([sample_restaurant])
    assert docs[0].metadata["cuisine"] == sample_restaurant["food_style"]


def test_build_article_docs_metadata_has_location(sample_restaurant):
    docs = _build_article_docs([sample_restaurant])
    assert docs[0].metadata["location"] == sample_restaurant["location"]


def test_build_article_docs_multiple_restaurants(sample_restaurants):
    extra = {**sample_restaurants[0], "name": "Second Place"}
    docs = _build_article_docs([sample_restaurants[0], extra])
    assert len(docs) == 2
    assert docs[0].metadata["doc_id"] != docs[1].metadata["doc_id"]


# ---------------------------------------------------------------------------
# _build_image_docs
# ---------------------------------------------------------------------------

def test_build_image_docs_creates_document(sample_recipe):
    docs = _build_image_docs(["images/recipe1.png"], [sample_recipe])
    assert len(docs) == 1
    assert "Grilled Salmon" in docs[0].page_content


def test_build_image_docs_metadata_has_image_path(sample_recipe):
    docs = _build_image_docs(["images/recipe1.png"], [sample_recipe])
    assert docs[0].metadata["image_path"] == "images/recipe1.png"


def test_build_image_docs_metadata_has_source(sample_recipe):
    docs = _build_image_docs(["images/r.png"], [sample_recipe])
    assert docs[0].metadata["source"] == "recipe_image"


def test_build_image_docs_stops_at_shorter_list(sample_recipe):
    # 3 paths but only 1 recipe → only 1 doc
    docs = _build_image_docs(["a.png", "b.png", "c.png"], [sample_recipe])
    assert len(docs) == 1


def test_build_image_docs_empty_inputs():
    docs = _build_image_docs([], [])
    assert docs == []


def test_build_image_docs_doc_id_format(sample_recipe):
    docs = _build_image_docs(["r.png"], [sample_recipe])
    assert docs[0].metadata["doc_id"].startswith("img_")


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

def _make_chroma_mock():
    mock = MagicMock()
    mock._collection = MagicMock()
    return mock


@patch("connoisseur.retrieval.build_index.shutil.rmtree")
@patch("connoisseur.retrieval.build_index.os.path.isdir", return_value=True)
@patch("connoisseur.retrieval.build_index.glob.glob", return_value=[])
@patch("connoisseur.retrieval.build_index.embed_images")
@patch("connoisseur.retrieval.build_index.embed_texts")
@patch("connoisseur.retrieval.build_index.Chroma")
def test_build_index_calls_rmtree_when_reset(
    MockChroma, mock_text_emb, mock_img_emb, mock_glob, mock_isdir, mock_rmtree,
    tmp_path, sample_restaurants, sample_recipes, restaurants_json, recipes_json
):
    mock_text_emb.return_value = np.ones((1, 384), dtype=np.float32)
    mock_img_emb.return_value = np.ones((0, 512), dtype=np.float32)
    MockChroma.return_value = _make_chroma_mock()

    with (
        patch("connoisseur.retrieval.build_index.RESTAURANT_DATA_PATH", restaurants_json),
        patch("connoisseur.retrieval.build_index.RECIPE_DATA_PATH", recipes_json),
        patch("connoisseur.retrieval.build_index.CHROMA_DIR", tmp_path / "chroma"),
        patch("connoisseur.retrieval.build_index.IMAGE_DIR", tmp_path / "images"),
    ):
        build_index(reset=True)

    mock_rmtree.assert_called_once()


@patch("connoisseur.retrieval.build_index.shutil.rmtree")
@patch("connoisseur.retrieval.build_index.os.path.isdir", return_value=True)
@patch("connoisseur.retrieval.build_index.glob.glob", return_value=[])
@patch("connoisseur.retrieval.build_index.embed_images")
@patch("connoisseur.retrieval.build_index.embed_texts")
@patch("connoisseur.retrieval.build_index.Chroma")
def test_build_index_skips_rmtree_when_no_reset(
    MockChroma, mock_text_emb, mock_img_emb, mock_glob, mock_isdir, mock_rmtree,
    tmp_path, restaurants_json, recipes_json
):
    mock_text_emb.return_value = np.ones((1, 384), dtype=np.float32)
    MockChroma.return_value = _make_chroma_mock()

    with (
        patch("connoisseur.retrieval.build_index.RESTAURANT_DATA_PATH", restaurants_json),
        patch("connoisseur.retrieval.build_index.RECIPE_DATA_PATH", recipes_json),
        patch("connoisseur.retrieval.build_index.CHROMA_DIR", tmp_path / "chroma"),
        patch("connoisseur.retrieval.build_index.IMAGE_DIR", tmp_path / "images"),
    ):
        build_index(reset=False)

    mock_rmtree.assert_not_called()


@patch("connoisseur.retrieval.build_index.shutil.rmtree")
@patch("connoisseur.retrieval.build_index.os.path.isdir", return_value=False)
@patch("connoisseur.retrieval.build_index.glob.glob", return_value=[])
@patch("connoisseur.retrieval.build_index.embed_images")
@patch("connoisseur.retrieval.build_index.embed_texts")
@patch("connoisseur.retrieval.build_index.Chroma")
def test_build_index_no_images_does_not_call_embed_images(
    MockChroma, mock_text_emb, mock_img_emb, mock_glob, mock_isdir, mock_rmtree,
    tmp_path, restaurants_json, recipes_json, capsys
):
    mock_text_emb.return_value = np.ones((1, 384), dtype=np.float32)
    MockChroma.return_value = _make_chroma_mock()

    with (
        patch("connoisseur.retrieval.build_index.RESTAURANT_DATA_PATH", restaurants_json),
        patch("connoisseur.retrieval.build_index.RECIPE_DATA_PATH", recipes_json),
        patch("connoisseur.retrieval.build_index.CHROMA_DIR", tmp_path / "chroma"),
        patch("connoisseur.retrieval.build_index.IMAGE_DIR", tmp_path / "images"),
    ):
        build_index(reset=False)

    mock_img_emb.assert_not_called()
    captured = capsys.readouterr()
    assert "No images" in captured.out


@patch("connoisseur.retrieval.build_index.shutil.rmtree")
@patch("connoisseur.retrieval.build_index.os.path.isdir", return_value=False)
@patch("connoisseur.retrieval.build_index.embed_images")
@patch("connoisseur.retrieval.build_index.embed_texts")
@patch("connoisseur.retrieval.build_index.Chroma")
def test_build_index_returns_two_chroma_instances(
    MockChroma, mock_text_emb, mock_img_emb, mock_rmtree,
    tmp_path, restaurants_json, recipes_json
):
    mock_text_emb.return_value = np.ones((1, 384), dtype=np.float32)
    img_file = tmp_path / "images" / "recipe1.png"
    img_file.parent.mkdir()
    img_file.write_bytes(b"\x89PNG")
    mock_img_emb.return_value = np.ones((1, 512), dtype=np.float32)

    chroma_a = _make_chroma_mock()
    chroma_b = _make_chroma_mock()
    MockChroma.side_effect = [chroma_a, chroma_b]

    with (
        patch("connoisseur.retrieval.build_index.RESTAURANT_DATA_PATH", restaurants_json),
        patch("connoisseur.retrieval.build_index.RECIPE_DATA_PATH", recipes_json),
        patch("connoisseur.retrieval.build_index.CHROMA_DIR", tmp_path / "chroma"),
        patch("connoisseur.retrieval.build_index.IMAGE_DIR", tmp_path / "images"),
    ):
        article_db, image_db = build_index(reset=False)

    assert article_db is chroma_a
    assert image_db is chroma_b
