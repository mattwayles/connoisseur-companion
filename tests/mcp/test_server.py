"""Tests for connoisseur.mcp.server — MCP tools and resource."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from connoisseur.mcp.server import (
    get_culinary_map,
    get_restaurant_info,
    recommend_by_vibe,
    get_review,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def restaurants():
    return [
        {
            "name": "The Gilded Artichoke",
            "location": "Beverly Hills",
            "food_style": "Modern American",
            "rating": 4.8,
            "vibe": "luxurious and romantic",
            "environment": "opulent dining room",
            "price_range": 4,
            "itemId": 1,
        },
        {
            "name": "Iron Chef Kitchen",
            "location": "DTLA",
            "food_style": "Japanese Fusion",
            "rating": 4.5,
            "vibe": "moody and sophisticated",
            "environment": "industrial chic",
            "price_range": 3,
            "itemId": 2,
        },
    ]


@pytest.fixture
def reviews(restaurants):
    return [
        {
            "itemId": 1,
            "userId": "user_a",
            "rating": 5.0,
            "title": "Best meal of my life",
            "text": "The Gilded Artichoke exceeded all expectations.",
            "image_captions": ["Beautifully plated dish"],
            "date": "2024-03-10",
        },
    ]


# ---------------------------------------------------------------------------
# get_culinary_map
# ---------------------------------------------------------------------------

def test_get_culinary_map_returns_text(tmp_path):
    map_file = tmp_path / "map.txt"
    map_file.write_text("California Culinary Map content here.")

    with patch("connoisseur.mcp.server.CULINARY_MAP_PATH", map_file):
        result = get_culinary_map()

    assert result == "California Culinary Map content here."


def test_get_culinary_map_returns_full_content(tmp_path):
    long_text = "Restaurant A.\n\nRestaurant B.\n\nRestaurant C."
    map_file = tmp_path / "map.txt"
    map_file.write_text(long_text)

    with patch("connoisseur.mcp.server.CULINARY_MAP_PATH", map_file):
        result = get_culinary_map()

    assert "Restaurant A" in result and "Restaurant C" in result


# ---------------------------------------------------------------------------
# get_restaurant_info
# ---------------------------------------------------------------------------

@patch("connoisseur.mcp.server._load_restaurants")
def test_get_restaurant_info_exact_match(mock_load, restaurants):
    mock_load.return_value = restaurants
    result = json.loads(get_restaurant_info("The Gilded Artichoke"))
    assert result["status"] == "found"
    assert result["count"] == 1
    assert result["results"][0]["name"] == "The Gilded Artichoke"


@patch("connoisseur.mcp.server._load_restaurants")
def test_get_restaurant_info_partial_match(mock_load, restaurants):
    mock_load.return_value = restaurants
    result = json.loads(get_restaurant_info("Gilded"))
    assert result["status"] == "found"
    assert result["count"] >= 1


@patch("connoisseur.mcp.server._load_restaurants")
def test_get_restaurant_info_case_insensitive(mock_load, restaurants):
    mock_load.return_value = restaurants
    result = json.loads(get_restaurant_info("iron chef"))
    assert result["status"] == "found"


@patch("connoisseur.mcp.server._load_restaurants")
def test_get_restaurant_info_not_found(mock_load, restaurants):
    mock_load.return_value = restaurants
    result = json.loads(get_restaurant_info("Nonexistent Place XYZ"))
    assert result["status"] == "not_found"
    assert "suggestion" in result


@patch("connoisseur.mcp.server._load_restaurants")
def test_get_restaurant_info_empty_database(mock_load):
    mock_load.return_value = []
    result = json.loads(get_restaurant_info("anything"))
    assert result["status"] == "not_found"


@patch("connoisseur.mcp.server._load_restaurants")
def test_get_restaurant_info_multiple_matches(mock_load, restaurants):
    extra = {**restaurants[0], "name": "The Gilded Spoon"}
    mock_load.return_value = restaurants + [extra]
    result = json.loads(get_restaurant_info("Gilded"))
    assert result["count"] == 2


# ---------------------------------------------------------------------------
# recommend_by_vibe
# ---------------------------------------------------------------------------

@patch("connoisseur.mcp.server.CULINARY_MAP_PATH")
@patch("connoisseur.mcp.server._load_restaurants")
def test_recommend_by_vibe_finds_structured_match(mock_load, mock_path, restaurants):
    mock_load.return_value = restaurants
    mock_path.read_text.return_value = ""

    result = json.loads(recommend_by_vibe("romantic"))
    assert result["vibe_searched"] == "romantic"
    assert len(result["structured_matches"]) >= 1
    assert result["structured_matches"][0]["name"] == "The Gilded Artichoke"


@patch("connoisseur.mcp.server.CULINARY_MAP_PATH")
@patch("connoisseur.mcp.server._load_restaurants")
def test_recommend_by_vibe_finds_raw_text_excerpts(mock_load, mock_path, restaurants):
    mock_load.return_value = []
    mock_path.read_text.return_value = (
        "First paragraph about zen atmosphere.\n\n"
        "Second paragraph about something else.\n\n"
        "Third paragraph with zen gardens.\n\n"
    )

    result = json.loads(recommend_by_vibe("zen"))
    assert len(result["raw_text_excerpts"]) >= 1


@patch("connoisseur.mcp.server.CULINARY_MAP_PATH")
@patch("connoisseur.mcp.server._load_restaurants")
def test_recommend_by_vibe_no_matches(mock_load, mock_path, restaurants):
    mock_load.return_value = restaurants
    mock_path.read_text.return_value = "Nothing relevant here."

    result = json.loads(recommend_by_vibe("nonexistent_vibe_xyz"))
    assert result["structured_matches"] == []
    assert result["raw_text_excerpts"] == []


@patch("connoisseur.mcp.server.CULINARY_MAP_PATH")
@patch("connoisseur.mcp.server._load_restaurants")
def test_recommend_by_vibe_case_insensitive(mock_load, mock_path, restaurants):
    mock_load.return_value = restaurants
    mock_path.read_text.return_value = ""

    result = json.loads(recommend_by_vibe("MOODY"))
    assert result["vibe_searched"] == "MOODY"
    # "moody" is in Iron Chef Kitchen's vibe field
    assert len(result["structured_matches"]) >= 1


@patch("connoisseur.mcp.server.CULINARY_MAP_PATH")
@patch("connoisseur.mcp.server._load_restaurants")
def test_recommend_by_vibe_result_has_required_keys(mock_load, mock_path, restaurants):
    mock_load.return_value = restaurants
    mock_path.read_text.return_value = ""

    result = json.loads(recommend_by_vibe("luxurious"))
    assert "vibe_searched" in result
    assert "structured_matches" in result
    assert "raw_text_excerpts" in result


# ---------------------------------------------------------------------------
# get_review
# ---------------------------------------------------------------------------

@patch("connoisseur.mcp.server._load_reviews")
@patch("connoisseur.mcp.server._load_restaurants")
def test_get_review_finds_by_restaurant_name(mock_load_rest, mock_load_rev, restaurants, reviews):
    mock_load_rest.return_value = restaurants
    mock_load_rev.return_value = reviews

    result = json.loads(get_review("Gilded Artichoke"))
    assert result["status"] == "found"
    assert "Gilded" in result["restaurant"]


@patch("connoisseur.mcp.server._load_reviews")
@patch("connoisseur.mcp.server._load_restaurants")
def test_get_review_finds_by_review_text(mock_load_rest, mock_load_rev, restaurants, reviews):
    mock_load_rest.return_value = restaurants
    mock_load_rev.return_value = reviews

    result = json.loads(get_review("exceeded all expectations"))
    assert result["status"] == "found"


@patch("connoisseur.mcp.server._load_reviews")
@patch("connoisseur.mcp.server._load_restaurants")
def test_get_review_finds_by_title(mock_load_rest, mock_load_rev, restaurants, reviews):
    mock_load_rest.return_value = restaurants
    mock_load_rev.return_value = reviews

    result = json.loads(get_review("Best meal"))
    assert result["status"] == "found"
    assert result["title"] == "Best meal of my life"


@patch("connoisseur.mcp.server._load_reviews")
@patch("connoisseur.mcp.server._load_restaurants")
def test_get_review_not_found(mock_load_rest, mock_load_rev, restaurants, reviews):
    mock_load_rest.return_value = restaurants
    mock_load_rev.return_value = reviews

    result = json.loads(get_review("XYZ Nonexistent Restaurant"))
    assert result["status"] == "not_found"


@patch("connoisseur.mcp.server._load_reviews")
@patch("connoisseur.mcp.server._load_restaurants")
def test_get_review_result_has_required_keys(mock_load_rest, mock_load_rev, restaurants, reviews):
    mock_load_rest.return_value = restaurants
    mock_load_rev.return_value = reviews

    result = json.loads(get_review("Gilded"))
    assert "status" in result
    assert "restaurant" in result
    assert "reviewer" in result
    assert "rating" in result
    assert "review_text" in result


@patch("connoisseur.mcp.server._load_reviews")
@patch("connoisseur.mcp.server._load_restaurants")
def test_get_review_empty_database(mock_load_rest, mock_load_rev):
    mock_load_rest.return_value = []
    mock_load_rev.return_value = []

    result = json.loads(get_review("anything"))
    assert result["status"] == "not_found"
