"""Shared pytest fixtures for the connoisseur-companion test suite."""
import json
import pytest

# ---------------------------------------------------------------------------
# Canonical sample data reused across test modules
# ---------------------------------------------------------------------------

SAMPLE_RESTAURANT = {
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": ["beer-battered snapper tacos", "zesty octopus ceviche"],
    "vibe": "salt-air energy",
    "environment": "open-air dining near the pier",
    "shortcomings": [],
    "itemId": 1000001,
}

SAMPLE_RECIPE = {
    "id": 1,
    "name": "Grilled Salmon",
    "cuisine": "American",
    "difficulty": "Easy",
    "prep_time": "30 minutes",
    "description": "A simple grilled salmon.",
}

SAMPLE_REVIEW = {
    "itemId": 1000001,
    "userId": "user123",
    "rating": 4.5,
    "title": "Amazing tacos!",
    "text": "The snapper tacos were outstanding.",
    "images": "[]",
    "date": "2024-01-15",
}


@pytest.fixture
def sample_restaurant():
    return dict(SAMPLE_RESTAURANT)


@pytest.fixture
def sample_restaurants():
    return [dict(SAMPLE_RESTAURANT)]


@pytest.fixture
def sample_recipes():
    return [dict(SAMPLE_RECIPE)]


@pytest.fixture
def sample_reviews():
    return [dict(SAMPLE_REVIEW)]


@pytest.fixture
def restaurants_json(tmp_path, sample_restaurants):
    """Write sample restaurants to a temp JSON file and return the path."""
    p = tmp_path / "restaurants.json"
    p.write_text(json.dumps(sample_restaurants))
    return p


@pytest.fixture
def recipes_json(tmp_path, sample_recipes):
    """Write sample recipes to a temp JSON file and return the path."""
    p = tmp_path / "recipes.json"
    p.write_text(json.dumps(sample_recipes))
    return p


@pytest.fixture
def reviews_json(tmp_path, sample_reviews):
    """Write sample reviews to a temp JSON file and return the path."""
    p = tmp_path / "reviews.json"
    p.write_text(json.dumps(sample_reviews))
    return p
