"""Tests for connoisseur.ingestion.structure_text."""
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pydantic import ValidationError

from connoisseur.ingestion.structure_text import (
    restaurant_data_structure_prompt,
    json_auto_repair_prompt,
    load_raw_paragraphs,
    structure_paragraph,
    run_structuring,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_RESTAURANT_JSON = json.dumps({
    "name": "Test Place",
    "location": "LA",
    "type": "bistro",
    "food_style": "French",
    "environment": "cozy indoor dining",
    "rating": 4.0,
    "price_range": 2,
    "signatures": [],
    "shortcomings": [],
})

CULINARY_MAP_TEXT = (
    "California Culinary Guide 2024\n\n"
    "Down in Santa Monica, Mar de Cortez serves fresh Baja seafood.\n\n"
    "In DTLA, The Copper Sprout offers farm-to-table Modern Appalachian.\n\n"
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_structure_prompt_returns_tuple():
    result = restaurant_data_structure_prompt("Some restaurant text.")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_structure_prompt_contains_paragraph():
    paragraph = "Amazing sushi in Malibu."
    _, user_prompt = restaurant_data_structure_prompt(paragraph)
    assert paragraph in user_prompt


def test_structure_prompt_system_is_nonempty():
    system, _ = restaurant_data_structure_prompt("text")
    assert system and isinstance(system, str)


def test_repair_prompt_returns_tuple():
    result = json_auto_repair_prompt('{"bad": }', "SyntaxError on line 1")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_repair_prompt_contains_candidate():
    candidate = '{"bad": }'
    _, user_prompt = json_auto_repair_prompt(candidate, "error msg")
    assert candidate in user_prompt


def test_repair_prompt_contains_error():
    error = "Field 'name' is required"
    _, user_prompt = json_auto_repair_prompt("{}", error)
    assert error in user_prompt


def test_repair_prompt_system_is_nonempty():
    system, _ = json_auto_repair_prompt("{}", "err")
    assert system and isinstance(system, str)


# ---------------------------------------------------------------------------
# load_raw_paragraphs
# ---------------------------------------------------------------------------

def test_load_raw_paragraphs_skips_header(tmp_path):
    map_file = tmp_path / "map.txt"
    map_file.write_text(CULINARY_MAP_TEXT)

    with patch("connoisseur.ingestion.structure_text.CULINARY_MAP_PATH", map_file):
        paragraphs = load_raw_paragraphs()

    assert len(paragraphs) == 2
    assert "Mar de Cortez" in paragraphs[0]
    assert "Copper Sprout" in paragraphs[1]


def test_load_raw_paragraphs_strips_whitespace(tmp_path):
    map_file = tmp_path / "map.txt"
    map_file.write_text("Header\n\n  Paragraph with spaces  \n\n")

    with patch("connoisseur.ingestion.structure_text.CULINARY_MAP_PATH", map_file):
        paragraphs = load_raw_paragraphs()

    assert paragraphs[0] == "Paragraph with spaces"


def test_load_raw_paragraphs_ignores_empty_sections(tmp_path):
    map_file = tmp_path / "map.txt"
    map_file.write_text("Header\n\n\n\nActual content.\n\n")

    with patch("connoisseur.ingestion.structure_text.CULINARY_MAP_PATH", map_file):
        paragraphs = load_raw_paragraphs()

    assert len(paragraphs) == 1
    assert paragraphs[0] == "Actual content."


def test_load_raw_paragraphs_header_only(tmp_path):
    map_file = tmp_path / "map.txt"
    map_file.write_text("Only a header line\n\n")

    with patch("connoisseur.ingestion.structure_text.CULINARY_MAP_PATH", map_file):
        paragraphs = load_raw_paragraphs()

    assert paragraphs == []


# ---------------------------------------------------------------------------
# structure_paragraph
# ---------------------------------------------------------------------------

@patch("connoisseur.ingestion.structure_text.call_llm")
def test_structure_paragraph_valid_on_first_attempt(mock_llm):
    mock_llm.return_value = VALID_RESTAURANT_JSON

    result = structure_paragraph("Some restaurant description.")

    assert result == VALID_RESTAURANT_JSON
    mock_llm.assert_called_once()


@patch("connoisseur.ingestion.structure_text.time.sleep")
@patch("connoisseur.ingestion.structure_text.call_llm")
def test_structure_paragraph_retries_on_validation_failure(mock_llm, mock_sleep):
    bad_json = json.dumps({"name": "X"})  # missing required fields → ValidationError
    mock_llm.side_effect = [bad_json, bad_json, VALID_RESTAURANT_JSON]

    result = structure_paragraph("Some text.")

    assert mock_llm.call_count == 3
    assert result == VALID_RESTAURANT_JSON


@patch("connoisseur.ingestion.structure_text.time.sleep")
@patch("connoisseur.ingestion.structure_text.call_llm")
def test_structure_paragraph_returns_best_effort_after_max_retries(mock_llm, mock_sleep):
    bad_json = json.dumps({"name": "X"})  # always invalid
    mock_llm.return_value = bad_json

    result = structure_paragraph("Some text.")

    # 1 initial + 3 repair attempts = 4 total LLM calls, but loop only retries 3 times
    assert mock_llm.call_count == 4
    assert result == bad_json  # returns whatever we have


@patch("connoisseur.ingestion.structure_text.time.sleep")
@patch("connoisseur.ingestion.structure_text.call_llm")
def test_structure_paragraph_handles_malformed_json(mock_llm, mock_sleep):
    mock_llm.side_effect = ["not json at all", "still bad", VALID_RESTAURANT_JSON]

    result = structure_paragraph("Some text.")

    assert result == VALID_RESTAURANT_JSON


# ---------------------------------------------------------------------------
# run_structuring
# ---------------------------------------------------------------------------

@patch("connoisseur.ingestion.structure_text.time.sleep")
@patch("connoisseur.ingestion.structure_text.call_llm")
def test_run_structuring_saves_records(mock_llm, mock_sleep, tmp_path):
    map_file = tmp_path / "map.txt"
    map_file.write_text("Header\n\nRestaurant A text.\n\nRestaurant B text.\n\n")
    out_file = tmp_path / "structured.json"
    mock_llm.return_value = VALID_RESTAURANT_JSON

    with (
        patch("connoisseur.ingestion.structure_text.CULINARY_MAP_PATH", map_file),
        patch("connoisseur.ingestion.structure_text.RESTAURANT_DATA_PATH", out_file),
    ):
        records = run_structuring(verbose=False)

    assert len(records) == 2
    assert out_file.exists()
    saved = json.loads(out_file.read_text())
    assert len(saved) == 2


@patch("connoisseur.ingestion.structure_text.time.sleep")
@patch("connoisseur.ingestion.structure_text.call_llm")
def test_run_structuring_assigns_item_ids(mock_llm, mock_sleep, tmp_path):
    map_file = tmp_path / "map.txt"
    map_file.write_text("Header\n\nRestaurant A.\n\nRestaurant B.\n\n")
    out_file = tmp_path / "structured.json"
    mock_llm.return_value = VALID_RESTAURANT_JSON

    with (
        patch("connoisseur.ingestion.structure_text.CULINARY_MAP_PATH", map_file),
        patch("connoisseur.ingestion.structure_text.RESTAURANT_DATA_PATH", out_file),
    ):
        records = run_structuring(verbose=False)

    item_ids = [r["itemId"] for r in records]
    assert item_ids[0] != item_ids[1]
    assert all(isinstance(i, int) for i in item_ids)


@patch("connoisseur.ingestion.structure_text.time.sleep")
@patch("connoisseur.ingestion.structure_text.call_llm")
def test_run_structuring_handles_unparseable_json(mock_llm, mock_sleep, tmp_path):
    map_file = tmp_path / "map.txt"
    map_file.write_text("Header\n\nSome restaurant.\n\n")
    out_file = tmp_path / "structured.json"
    # Always returns non-JSON
    mock_llm.return_value = "not json"

    with (
        patch("connoisseur.ingestion.structure_text.CULINARY_MAP_PATH", map_file),
        patch("connoisseur.ingestion.structure_text.RESTAURANT_DATA_PATH", out_file),
    ):
        records = run_structuring(verbose=False)

    # Falls back to a placeholder record
    assert len(records) == 1
    assert "itemId" in records[0]
