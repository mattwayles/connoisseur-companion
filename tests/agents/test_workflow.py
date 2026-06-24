"""Tests for connoisseur.agents.workflow — agent orchestration and JSON parsing."""
import json
import pytest
from unittest.mock import MagicMock, patch

from connoisseur.agents.workflow import (
    _make_system,
    _parse_json,
    node_generate_profile,
    node_retrieve_candidates,
    node_analyze_trends,
    node_analyze_styles,
    node_evaluate_nutrition,
    node_generate_recommendations,
    run_workflow,
)


# ---------------------------------------------------------------------------
# _make_system
# ---------------------------------------------------------------------------

def test_make_system_contains_role():
    system = _make_system("user_profile_generator")
    assert "User Profile Generator" in system


def test_make_system_contains_goal():
    system = _make_system("rag_retriever")
    assert "goal" in system.lower() or "vector" in system.lower()


def test_make_system_contains_backstory():
    system = _make_system("food_trend_analyst")
    assert "culinary journalist" in system.lower() or "trend" in system.lower()


def test_make_system_instructs_json_output():
    system = _make_system("recommendation_expert")
    assert "JSON" in system


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

def test_parse_json_valid_dict():
    result = _parse_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_valid_list():
    result = _parse_json('[1, 2, 3]')
    assert result == [1, 2, 3]


def test_parse_json_strips_whitespace():
    result = _parse_json('  {"a": 1}  ')
    assert result == {"a": 1}


def test_parse_json_fenced_with_closing_backticks():
    text = "```json\n{\"x\": 42}\n```"
    result = _parse_json(text)
    assert result == {"x": 42}


def test_parse_json_fenced_without_language_specifier():
    text = "```\n{\"x\": 1}\n```"
    result = _parse_json(text)
    assert result == {"x": 1}


def test_parse_json_fenced_without_closing_backticks():
    text = "```json\n{\"x\": 99}"
    result = _parse_json(text)
    assert result == {"x": 99}


def test_parse_json_fenced_multiline_json():
    text = "```json\n{\n  \"name\": \"test\",\n  \"val\": 1\n}\n```"
    result = _parse_json(text)
    assert result == {"name": "test", "val": 1}


def test_parse_json_invalid_returns_raw():
    result = _parse_json("not json at all")
    assert result == {"raw": "not json at all"}


def test_parse_json_empty_string_returns_raw():
    result = _parse_json("")
    assert "raw" in result


def test_parse_json_empty_fenced_block():
    text = "```json\n```"
    result = _parse_json(text)
    assert "raw" in result  # empty content fails JSON parse


def test_parse_json_nested_structure():
    data = {"restaurants": [{"name": "X", "rating": 4.5}], "recipes": []}
    result = _parse_json(json.dumps(data))
    assert result["restaurants"][0]["name"] == "X"


# ---------------------------------------------------------------------------
# Phase nodes
# ---------------------------------------------------------------------------

def _base_state():
    return {
        "user_input": "I love spicy food",
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start",
    }


PROFILE_JSON = json.dumps({
    "favorite_cuisines": ["Thai"],
    "dietary_restrictions": [],
    "dining_occasions": ["casual"],
    "price_range": "$$",
    "adventurousness_score": 8,
    "flavor_preferences": ["spicy"],
    "summary": "Loves spicy Thai food",
})

CANDIDATES_JSON = json.dumps({
    "restaurants": [{"name": "Bangkok Garden", "cuisine": "Thai", "price": "$$", "rating": 4.5, "description": "Authentic Thai."}],
    "recipes": [{"name": "Pad Thai", "cuisine": "Thai", "difficulty": "Medium", "prep_time": "30m", "description": "Classic."}],
})

TRENDS_JSON = json.dumps({
    "trends": [{"name": "Plant-based Thai", "description": "...", "relevance": "high"}],
})

STYLES_JSON = json.dumps({
    "cuisines": [{"name": "Thai", "description": "Southeast Asian"}],
    "profiles": [{"name": "Spicy", "description": "High heat"}],
})

NUTRITION_JSON = json.dumps({
    "compliant_items": ["Pad Thai"],
    "flagged_items": [],
    "nutritional_highlights": ["High protein"],
})

RECS_JSON = json.dumps({
    "restaurants": [{"name": "Bangkok Garden", "reasoning": "Great Thai food that matches your love of spice."}],
    "recipes": [{"name": "Pad Thai", "reasoning": "Classic spicy noodles perfect for you."}],
})


@patch("connoisseur.agents.workflow.call_llm", return_value=PROFILE_JSON)
def test_node_generate_profile_updates_state(mock_llm):
    state = _base_state()
    result = node_generate_profile(state)
    assert result["user_profile"]["summary"] == "Loves spicy Thai food"
    assert result["workflow_step"] == "profile_generated"


@patch("connoisseur.agents.workflow.call_llm", return_value=CANDIDATES_JSON)
def test_node_retrieve_candidates_updates_state(mock_llm):
    state = _base_state()
    state["user_profile"] = json.loads(PROFILE_JSON)
    result = node_retrieve_candidates(state)
    assert len(result["retrieved_restaurants"]) == 1
    assert len(result["retrieved_recipes"]) == 1
    assert result["workflow_step"] == "candidates_retrieved"


@patch("connoisseur.agents.workflow.call_llm", return_value=TRENDS_JSON)
def test_node_analyze_trends_updates_state(mock_llm):
    state = _base_state()
    state["retrieved_restaurants"] = [{"name": "Test"}]
    state["retrieved_recipes"] = [{"name": "Recipe"}]
    result = node_analyze_trends(state)
    assert "trends" in result["trend_analysis"]


@patch("connoisseur.agents.workflow.call_llm", return_value=STYLES_JSON)
def test_node_analyze_styles_updates_state(mock_llm):
    state = _base_state()
    state["retrieved_restaurants"] = [{"name": "Test"}]
    state["retrieved_recipes"] = [{"name": "Recipe"}]
    result = node_analyze_styles(state)
    assert "cuisines" in result["style_analysis"]


@patch("connoisseur.agents.workflow.call_llm", return_value=NUTRITION_JSON)
def test_node_evaluate_nutrition_updates_state(mock_llm):
    state = _base_state()
    state["user_profile"] = json.loads(PROFILE_JSON)
    state["retrieved_restaurants"] = [{"name": "Test"}]
    state["retrieved_recipes"] = [{"name": "Recipe"}]
    result = node_evaluate_nutrition(state)
    assert "compliant_items" in result["nutrition_analysis"]


@patch("connoisseur.agents.workflow.call_llm", return_value=RECS_JSON)
def test_node_generate_recommendations_updates_state(mock_llm):
    state = _base_state()
    state["user_profile"] = json.loads(PROFILE_JSON)
    state["retrieved_restaurants"] = [{"name": "Test"}]
    state["retrieved_recipes"] = [{"name": "Recipe"}]
    state["trend_analysis"] = json.loads(TRENDS_JSON)
    state["style_analysis"] = json.loads(STYLES_JSON)
    state["nutrition_analysis"] = json.loads(NUTRITION_JSON)
    result = node_generate_recommendations(state)
    assert result["final_recommendations"]["restaurants"][0]["name"] == "Bangkok Garden"
    assert result["workflow_step"] == "complete"


# ---------------------------------------------------------------------------
# Error handling in nodes
# ---------------------------------------------------------------------------

@patch("connoisseur.agents.workflow.call_llm", side_effect=Exception("API error"))
def test_node_generate_profile_handles_exception(mock_llm):
    state = _base_state()
    result = node_generate_profile(state)
    assert "error" in result["user_profile"]


@patch("connoisseur.agents.workflow.call_llm", side_effect=Exception("API error"))
def test_node_retrieve_candidates_handles_exception(mock_llm):
    state = _base_state()
    state["user_profile"] = {}
    result = node_retrieve_candidates(state)
    assert result["retrieved_restaurants"] == []
    assert result["retrieved_recipes"] == []


@patch("connoisseur.agents.workflow.call_llm", side_effect=Exception("API error"))
def test_node_analyze_trends_handles_exception(mock_llm):
    state = _base_state()
    state["retrieved_restaurants"] = []
    state["retrieved_recipes"] = []
    result = node_analyze_trends(state)
    assert result["trend_analysis"].get("trends") == []


# ---------------------------------------------------------------------------
# run_workflow integration
# ---------------------------------------------------------------------------

@patch("connoisseur.agents.workflow.call_llm")
def test_run_workflow_returns_complete_state(mock_llm):
    mock_llm.side_effect = [
        PROFILE_JSON,
        CANDIDATES_JSON,
        TRENDS_JSON,
        STYLES_JSON,
        NUTRITION_JSON,
        RECS_JSON,
    ]
    result = run_workflow("I love spicy Thai food")
    assert result["workflow_step"] == "complete"
    assert "restaurants" in result["final_recommendations"]
    assert "recipes" in result["final_recommendations"]


@patch("connoisseur.agents.workflow.call_llm")
def test_run_workflow_all_phases_called(mock_llm):
    mock_llm.side_effect = [
        PROFILE_JSON,
        CANDIDATES_JSON,
        TRENDS_JSON,
        STYLES_JSON,
        NUTRITION_JSON,
        RECS_JSON,
    ]
    run_workflow("test input")
    # 1 profile + 1 candidates + 3 parallel + 1 recs = 6 calls
    assert mock_llm.call_count == 6
