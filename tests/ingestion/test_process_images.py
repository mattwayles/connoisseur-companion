"""Tests for connoisseur.ingestion.process_images."""
import base64
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from connoisseur.ingestion.process_images import (
    _b64_from_path,
    _b64_from_bytes,
    _recipe_caption_prompt,
    _review_caption_prompt,
    _fetch_url,
    caption_recipe_images,
    caption_review_images,
)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def test_b64_from_path_png(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG test bytes")

    b64, media_type = _b64_from_path(img)

    assert media_type == "image/png"
    assert base64.b64decode(b64) == b"\x89PNG test bytes"


def test_b64_from_path_jpg(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff jpeg bytes")

    _, media_type = _b64_from_path(img)

    assert media_type == "image/jpeg"


def test_b64_from_path_jpeg_extension(tmp_path):
    img = tmp_path / "photo.jpeg"
    img.write_bytes(b"data")

    _, media_type = _b64_from_path(img)

    assert media_type == "image/jpeg"


def test_b64_from_bytes_encodes_correctly():
    data = b"hello bytes"
    b64, media_type = _b64_from_bytes(data, "image/jpeg")

    assert media_type == "image/jpeg"
    assert base64.b64decode(b64) == data


def test_b64_from_bytes_default_media_type():
    _, media_type = _b64_from_bytes(b"data")
    assert media_type == "image/jpeg"


def test_b64_from_bytes_empty():
    b64, _ = _b64_from_bytes(b"")
    assert base64.b64decode(b64) == b""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_recipe_caption_prompt_returns_tuple():
    result = _recipe_caption_prompt("Grilled Salmon")
    assert isinstance(result, tuple) and len(result) == 2


def test_recipe_caption_prompt_contains_food_name():
    _, user_prompt = _recipe_caption_prompt("Grilled Salmon")
    assert "Grilled Salmon" in user_prompt


def test_recipe_caption_prompt_system_nonempty():
    system, _ = _recipe_caption_prompt("X")
    assert system


def test_review_caption_prompt_returns_tuple():
    result = _review_caption_prompt("The food was amazing.")
    assert isinstance(result, tuple) and len(result) == 2


def test_review_caption_prompt_contains_review_text():
    _, user_prompt = _review_caption_prompt("The food was amazing.")
    assert "The food was amazing." in user_prompt


def test_review_caption_prompt_system_nonempty():
    system, _ = _review_caption_prompt("review text")
    assert system


# ---------------------------------------------------------------------------
# _fetch_url
# ---------------------------------------------------------------------------

@patch("connoisseur.ingestion.process_images.requests.get")
def test_fetch_url_returns_content_on_success(mock_get):
    resp = MagicMock()
    resp.content = b"image bytes"
    resp.raise_for_status = MagicMock()
    mock_get.return_value = resp

    result = _fetch_url("https://example.com/image.jpg")

    assert result == b"image bytes"


@patch("connoisseur.ingestion.process_images.requests.get")
def test_fetch_url_raises_on_http_error(mock_get):
    import requests
    mock_get.side_effect = requests.HTTPError("404 Not Found")

    # Access the unwrapped function to bypass tenacity's retry logic
    unwrapped = getattr(_fetch_url, "__wrapped__", _fetch_url)
    with pytest.raises(requests.HTTPError):
        unwrapped("https://example.com/missing.jpg")


# ---------------------------------------------------------------------------
# caption_recipe_images
# ---------------------------------------------------------------------------

@patch("connoisseur.ingestion.process_images.call_vision_llm")
def test_caption_recipe_images_adds_description(mock_llm, tmp_path, sample_recipes):
    mock_llm.return_value = "A golden fillet on a white plate."
    recipes_file = tmp_path / "recipes.json"
    recipes_file.write_text(json.dumps(sample_recipes))

    # Create a fake PNG for recipe1
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "recipe1.png").write_bytes(b"\x89PNG")

    out_file = tmp_path / "augmented_recipes.json"

    with (
        patch("connoisseur.ingestion.process_images.RAW_RECIPES_PATH", recipes_file),
        patch("connoisseur.ingestion.process_images.IMAGE_DIR", image_dir),
        patch("connoisseur.ingestion.process_images.RECIPE_DATA_PATH", out_file),
    ):
        result = caption_recipe_images(verbose=False)

    assert result[0]["image_description"] == "A golden fillet on a white plate."
    assert out_file.exists()


@patch("connoisseur.ingestion.process_images.call_vision_llm")
def test_caption_recipe_images_handles_missing_image(mock_llm, tmp_path, sample_recipes):
    recipes_file = tmp_path / "recipes.json"
    recipes_file.write_text(json.dumps(sample_recipes))
    out_file = tmp_path / "augmented_recipes.json"

    # No image file created → empty description
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    with (
        patch("connoisseur.ingestion.process_images.RAW_RECIPES_PATH", recipes_file),
        patch("connoisseur.ingestion.process_images.IMAGE_DIR", image_dir),
        patch("connoisseur.ingestion.process_images.RECIPE_DATA_PATH", out_file),
    ):
        result = caption_recipe_images(verbose=False)

    assert result[0]["image_description"] == ""
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# caption_review_images
# ---------------------------------------------------------------------------

@patch("connoisseur.ingestion.process_images.call_vision_llm")
@patch("connoisseur.ingestion.process_images._fetch_url")
def test_caption_review_images_adds_captions(mock_fetch, mock_llm, tmp_path):
    mock_fetch.return_value = b"\xff\xd8\xff jpeg bytes"
    mock_llm.return_value = "Crispy tacos on a rustic board."

    review = {
        "itemId": 1,
        "text": "Great food!",
        "images": '["https://example.com/img.jpg"]',
    }
    reviews_file = tmp_path / "reviews.json"
    reviews_file.write_text(json.dumps([review]))
    out_file = tmp_path / "augmented_reviews.json"

    with (
        patch("connoisseur.ingestion.process_images.RAW_REVIEWS_PATH", reviews_file),
        patch("connoisseur.ingestion.process_images.REVIEW_DATA_PATH", out_file),
        patch("connoisseur.ingestion.process_images.DATA_DIR", tmp_path),
    ):
        result = caption_review_images(verbose=False)

    assert result[0]["image_captions"] == ["Crispy tacos on a rustic board."]
    assert out_file.exists()


@patch("connoisseur.ingestion.process_images.call_vision_llm")
@patch("connoisseur.ingestion.process_images._fetch_url")
def test_caption_review_images_handles_fetch_failure(mock_fetch, mock_llm, tmp_path):
    mock_fetch.side_effect = Exception("network error")

    review = {
        "itemId": 1,
        "text": "Review text.",
        "images": '["https://example.com/img.jpg"]',
    }
    reviews_file = tmp_path / "reviews.json"
    reviews_file.write_text(json.dumps([review]))
    out_file = tmp_path / "augmented_reviews.json"

    with (
        patch("connoisseur.ingestion.process_images.RAW_REVIEWS_PATH", reviews_file),
        patch("connoisseur.ingestion.process_images.REVIEW_DATA_PATH", out_file),
        patch("connoisseur.ingestion.process_images.DATA_DIR", tmp_path),
    ):
        result = caption_review_images(verbose=False)

    assert result[0]["image_captions"] == []
    mock_llm.assert_not_called()


@patch("connoisseur.ingestion.process_images.call_vision_llm")
@patch("connoisseur.ingestion.process_images._fetch_url")
def test_caption_review_images_empty_image_list(mock_fetch, mock_llm, tmp_path):
    review = {"itemId": 1, "text": "No photos.", "images": "[]"}
    reviews_file = tmp_path / "reviews.json"
    reviews_file.write_text(json.dumps([review]))
    out_file = tmp_path / "augmented_reviews.json"

    with (
        patch("connoisseur.ingestion.process_images.RAW_REVIEWS_PATH", reviews_file),
        patch("connoisseur.ingestion.process_images.REVIEW_DATA_PATH", out_file),
        patch("connoisseur.ingestion.process_images.DATA_DIR", tmp_path),
    ):
        result = caption_review_images(verbose=False)

    assert result[0]["image_captions"] == []
    mock_fetch.assert_not_called()
    mock_llm.assert_not_called()
