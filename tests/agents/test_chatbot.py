"""Tests for connoisseur.agents.chatbot — intent classification, preference extraction, formatting."""
import json
import pytest
from unittest.mock import MagicMock, patch

from connoisseur.agents.chatbot import (
    classify_intent,
    extract_preferences,
    format_recommendations,
    recommendation_chatbot,
)


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent", ["restaurant", "recipe", "both", "clarification", "database"])
@patch("connoisseur.agents.chatbot.call_llm")
def test_classify_intent_returns_valid_intent(mock_llm, intent):
    mock_llm.return_value = intent
    result = classify_intent("some message")
    assert result == intent


@patch("connoisseur.agents.chatbot.call_llm")
def test_classify_intent_falls_back_to_clarification_on_unknown(mock_llm):
    mock_llm.return_value = "something_totally_invalid"
    result = classify_intent("hello")
    assert result == "clarification"


@patch("connoisseur.agents.chatbot.call_llm")
def test_classify_intent_strips_whitespace(mock_llm):
    mock_llm.return_value = "  restaurant  "
    result = classify_intent("I want a restaurant")
    assert result == "restaurant"


@patch("connoisseur.agents.chatbot.call_llm")
def test_classify_intent_lowercases_result(mock_llm):
    mock_llm.return_value = "RECIPE"
    result = classify_intent("give me a recipe")
    assert result == "recipe"


@patch("connoisseur.agents.chatbot.call_llm")
def test_classify_intent_empty_response_gives_clarification(mock_llm):
    mock_llm.return_value = ""
    result = classify_intent("hi")
    assert result == "clarification"


# ---------------------------------------------------------------------------
# extract_preferences
# ---------------------------------------------------------------------------

PREFS_JSON = json.dumps({
    "favorite_cuisines": ["Thai", "Italian"],
    "dietary_restrictions": ["gluten-free"],
    "dining_occasion": "date night",
    "price_range": "$$$",
    "flavor_preferences": ["spicy"],
    "other_preferences": "outdoor seating",
})


@patch("connoisseur.agents.chatbot.call_llm")
def test_extract_preferences_parses_valid_json(mock_llm):
    mock_llm.return_value = PREFS_JSON
    prefs = extract_preferences("I want gluten-free Thai for a date night")
    assert prefs["favorite_cuisines"] == ["Thai", "Italian"]
    assert prefs["dietary_restrictions"] == ["gluten-free"]
    assert prefs["dining_occasion"] == "date night"


@patch("connoisseur.agents.chatbot.call_llm")
def test_extract_preferences_strips_markdown_fence(mock_llm):
    mock_llm.return_value = f"```json\n{PREFS_JSON}\n```"
    prefs = extract_preferences("some message")
    assert "favorite_cuisines" in prefs


@patch("connoisseur.agents.chatbot.call_llm")
def test_extract_preferences_returns_defaults_on_parse_error(mock_llm):
    mock_llm.return_value = "not valid json"
    prefs = extract_preferences("message")
    assert "favorite_cuisines" in prefs
    assert isinstance(prefs["favorite_cuisines"], list)
    assert prefs["dining_occasion"] == "not specified"
    assert prefs["price_range"] == "not specified"


@patch("connoisseur.agents.chatbot.call_llm")
def test_extract_preferences_returns_defaults_on_empty_response(mock_llm):
    mock_llm.return_value = ""
    prefs = extract_preferences("message")
    assert prefs["other_preferences"] == ""


# ---------------------------------------------------------------------------
# format_recommendations
# ---------------------------------------------------------------------------

def test_format_recommendations_with_restaurants():
    recs = {
        "restaurants": [{"name": "Bangkok Garden", "reasoning": "Perfect for spice lovers."}],
        "recipes": [],
    }
    result = format_recommendations(recs)
    assert "Bangkok Garden" in result
    assert "Perfect for spice lovers." in result
    assert "Restaurant" in result


def test_format_recommendations_with_recipes():
    recs = {
        "restaurants": [],
        "recipes": [{"name": "Pad Thai", "reasoning": "Classic noodle dish."}],
    }
    result = format_recommendations(recs)
    assert "Pad Thai" in result
    assert "Classic noodle dish." in result
    assert "Recipe" in result


def test_format_recommendations_with_both():
    recs = {
        "restaurants": [{"name": "Sakura", "reasoning": "Great sushi."}],
        "recipes": [{"name": "Miso Soup", "reasoning": "Warming and umami."}],
    }
    result = format_recommendations(recs)
    assert "Sakura" in result
    assert "Miso Soup" in result


def test_format_recommendations_empty_returns_fallback():
    result = format_recommendations({})
    assert "couldn't" in result.lower() or "try" in result.lower()


def test_format_recommendations_empty_lists_returns_fallback():
    result = format_recommendations({"restaurants": [], "recipes": []})
    assert "couldn't" in result.lower() or "try" in result.lower()


def test_format_recommendations_missing_name_shows_unknown():
    recs = {"restaurants": [{"reasoning": "No name provided."}], "recipes": []}
    result = format_recommendations(recs)
    assert "Unknown" in result


def test_format_recommendations_missing_reasoning_still_works():
    recs = {"restaurants": [{"name": "Place A"}], "recipes": []}
    result = format_recommendations(recs)
    assert "Place A" in result


def test_format_recommendations_multiple_restaurants():
    recs = {
        "restaurants": [
            {"name": "Place A", "reasoning": "First."},
            {"name": "Place B", "reasoning": "Second."},
        ],
        "recipes": [],
    }
    result = format_recommendations(recs)
    assert "Place A" in result
    assert "Place B" in result
    assert "1." in result
    assert "2." in result


# ---------------------------------------------------------------------------
# recommendation_chatbot
# ---------------------------------------------------------------------------

@patch("connoisseur.agents.chatbot.classify_intent", return_value="clarification")
def test_recommendation_chatbot_clarification_intent(mock_intent):
    result = recommendation_chatbot("hello", [])
    assert "Connoisseur Companion" in result


@patch("connoisseur.agents.chatbot.classify_intent", return_value="database")
def test_recommendation_chatbot_database_intent(mock_intent):
    result = recommendation_chatbot("add a restaurant", [])
    assert "database" in result.lower() or "Add Restaurant" in result


@patch("connoisseur.agents.chatbot.run_workflow")
@patch("connoisseur.agents.chatbot.extract_preferences")
@patch("connoisseur.agents.chatbot.classify_intent")
def test_recommendation_chatbot_restaurant_intent(mock_intent, mock_prefs, mock_workflow):
    mock_intent.return_value = "restaurant"
    mock_prefs.return_value = {
        "favorite_cuisines": ["Italian"],
        "dietary_restrictions": [],
        "dining_occasion": "dinner",
        "price_range": "$$",
        "flavor_preferences": [],
        "other_preferences": "",
    }
    mock_workflow.return_value = {
        "final_recommendations": {
            "restaurants": [{"name": "Roma", "reasoning": "Great Italian."}],
            "recipes": [],
        }
    }
    result = recommendation_chatbot("Italian restaurants please", [])
    assert "Roma" in result


@patch("connoisseur.agents.chatbot.run_workflow")
@patch("connoisseur.agents.chatbot.extract_preferences")
@patch("connoisseur.agents.chatbot.classify_intent")
def test_recommendation_chatbot_both_intent(mock_intent, mock_prefs, mock_workflow):
    mock_intent.return_value = "both"
    mock_prefs.return_value = {
        "favorite_cuisines": [],
        "dietary_restrictions": [],
        "dining_occasion": "not specified",
        "price_range": "any",
        "flavor_preferences": [],
        "other_preferences": "",
    }
    mock_workflow.return_value = {
        "final_recommendations": {
            "restaurants": [{"name": "Place X", "reasoning": "Good."}],
            "recipes": [{"name": "Dish Y", "reasoning": "Tasty."}],
        }
    }
    result = recommendation_chatbot("both restaurants and recipes", [])
    assert "Place X" in result
    assert "Dish Y" in result
