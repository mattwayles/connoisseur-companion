"""Tests for connoisseur.ingestion.models — Pydantic Restaurant schema."""
import json
import pytest
from pydantic import ValidationError
from connoisseur.ingestion.models import Restaurant


REQUIRED = {
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "environment": "open-air dining near the pier",
}


# ---------------------------------------------------------------------------
# Happy-path construction
# ---------------------------------------------------------------------------

def test_minimal_valid_restaurant():
    r = Restaurant(**REQUIRED)
    assert r.name == "Mar de Cortez"
    assert r.location == "Santa Monica"
    assert r.type == "casual taqueria"
    assert r.food_style == "Baja-style seafood"
    assert r.environment == "open-air dining near the pier"


def test_optional_fields_default_correctly():
    r = Restaurant(**REQUIRED)
    assert r.rating is None
    assert r.price_range is None
    assert r.signatures == []
    assert r.vibe is None
    assert r.shortcomings == []
    assert r.itemId is None


def test_full_restaurant_all_fields():
    data = {
        **REQUIRED,
        "rating": 4.2,
        "price_range": 2,
        "signatures": ["taco", "ceviche"],
        "vibe": "salt-air",
        "shortcomings": ["parking"],
        "itemId": 1000001,
    }
    r = Restaurant(**data)
    assert r.rating == 4.2
    assert r.price_range == 2
    assert r.signatures == ["taco", "ceviche"]
    assert r.vibe == "salt-air"
    assert r.shortcomings == ["parking"]
    assert r.itemId == 1000001


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_field", ["name", "location", "type", "food_style", "environment"])
def test_missing_required_field_raises(missing_field):
    data = {k: v for k, v in REQUIRED.items() if k != missing_field}
    with pytest.raises(ValidationError):
        Restaurant(**data)


# ---------------------------------------------------------------------------
# Type coercion and validation
# ---------------------------------------------------------------------------

def test_rating_int_coerced_to_float():
    r = Restaurant(**{**REQUIRED, "rating": 4})
    assert isinstance(r.rating, float)
    assert r.rating == 4.0


def test_rating_string_raises():
    with pytest.raises(ValidationError):
        Restaurant(**{**REQUIRED, "rating": "not-a-number"})


def test_price_range_accepts_none():
    r = Restaurant(**{**REQUIRED, "price_range": None})
    assert r.price_range is None


def test_price_range_accepts_integer():
    r = Restaurant(**{**REQUIRED, "price_range": 3})
    assert r.price_range == 3


def test_signatures_accepts_list():
    r = Restaurant(**{**REQUIRED, "signatures": ["a", "b"]})
    assert r.signatures == ["a", "b"]


def test_shortcomings_accepts_empty_list():
    r = Restaurant(**{**REQUIRED, "shortcomings": []})
    assert r.shortcomings == []


def test_item_id_accepts_none():
    r = Restaurant(**{**REQUIRED, "itemId": None})
    assert r.itemId is None


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_model_validate_json_round_trip():
    data = {**REQUIRED, "rating": 4.2, "itemId": 42}
    r = Restaurant.model_validate_json(json.dumps(data))
    assert r.name == "Mar de Cortez"
    assert r.itemId == 42


def test_model_validate_json_missing_required_raises():
    bad = {"name": "X", "location": "Y"}
    with pytest.raises(ValidationError):
        Restaurant.model_validate_json(json.dumps(bad))


def test_model_dump_round_trip():
    r = Restaurant(**{**REQUIRED, "rating": 3.9})
    dumped = r.model_dump()
    r2 = Restaurant(**dumped)
    assert r2.name == r.name
    assert r2.rating == r.rating
