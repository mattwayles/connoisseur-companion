"""Tests for connoisseur.ingestion.data_manager — CRUD helpers and CLI."""
import io
import json
import os
import pytest
from unittest.mock import MagicMock, patch

from connoisseur.ingestion.data_manager import (
    load_data,
    save_data,
    show_restaurant_card,
    new_data_entry_process,
    manage_restaurants,
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


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_returns_list(tmp_path):
    data = [{"name": "Cafe A"}, {"name": "Cafe B"}]
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data))

    result = load_data(str(f))

    assert result == data


def test_load_data_returns_empty_when_file_missing():
    result = load_data("/nonexistent/path/data.json")
    assert result == []


def test_load_data_returns_empty_on_malformed_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")

    result = load_data(str(f))

    assert result == []


def test_load_data_handles_empty_list(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text("[]")

    result = load_data(str(f))

    assert result == []


# ---------------------------------------------------------------------------
# save_data
# ---------------------------------------------------------------------------

def test_save_data_writes_json(tmp_path):
    data = [{"name": "Test"}]
    out = tmp_path / "out.json"
    bak = tmp_path / "out.json.bak"

    save_data(data, str(out), str(bak))

    saved = json.loads(out.read_text())
    assert saved == data


def test_save_data_creates_backup_when_original_exists(tmp_path):
    original = [{"name": "Old"}]
    out = tmp_path / "data.json"
    bak = tmp_path / "data.json.bak"
    out.write_text(json.dumps(original))

    save_data([{"name": "New"}], str(out), str(bak))

    assert bak.exists()
    backed_up = json.loads(bak.read_text())
    assert backed_up == original


def test_save_data_no_backup_when_original_absent(tmp_path):
    out = tmp_path / "data.json"
    bak = tmp_path / "data.json.bak"

    save_data([{"name": "First"}], str(out), str(bak))

    assert not bak.exists()
    assert out.exists()


def test_save_data_overwrites_existing(tmp_path):
    out = tmp_path / "data.json"
    bak = tmp_path / "data.json.bak"
    out.write_text(json.dumps([{"name": "Old"}]))

    save_data([{"name": "New"}], str(out), str(bak))

    saved = json.loads(out.read_text())
    assert saved[0]["name"] == "New"


# ---------------------------------------------------------------------------
# show_restaurant_card
# ---------------------------------------------------------------------------

def test_show_restaurant_card_displays_name(capsys):
    restaurant = {"name": "Sunset Grill", "location": "Malibu", "rating": 4.5}
    show_restaurant_card(restaurant, 0)
    captured = capsys.readouterr()
    assert "Sunset Grill" in captured.out


def test_show_restaurant_card_uses_restaurant_name_key(capsys):
    restaurant = {"restaurant_name": "Old Style Name", "location": "SF"}
    show_restaurant_card(restaurant, 1)
    captured = capsys.readouterr()
    assert "Old Style Name" in captured.out


def test_show_restaurant_card_shows_index(capsys):
    show_restaurant_card({"name": "X", "location": "Y"}, 42)
    captured = capsys.readouterr()
    assert "42" in captured.out


def test_show_restaurant_card_shows_all_fields(capsys):
    restaurant = {"name": "A", "location": "LA", "food_style": "Italian"}
    show_restaurant_card(restaurant, 0)
    captured = capsys.readouterr()
    assert "LOCATION" in captured.out
    assert "FOOD_STYLE" in captured.out or "FOOD STYLE" in captured.out


# ---------------------------------------------------------------------------
# new_data_entry_process
# ---------------------------------------------------------------------------

@patch("connoisseur.ingestion.data_manager.call_llm")
def test_new_data_entry_process_assigns_item_id(mock_llm):
    mock_llm.return_value = VALID_RESTAURANT_JSON

    record = new_data_entry_process("A nice bistro in LA.", 9999)

    assert record["itemId"] == 9999


@patch("connoisseur.ingestion.data_manager.time.sleep")
@patch("connoisseur.ingestion.data_manager.call_llm")
def test_new_data_entry_process_retries_on_validation_error(mock_llm, mock_sleep):
    bad = json.dumps({"name": "Only name"})  # missing required fields
    mock_llm.side_effect = [bad, bad, VALID_RESTAURANT_JSON]

    record = new_data_entry_process("A nice bistro in LA.", 1)

    assert record["name"] == "Test Place"
    assert mock_llm.call_count == 3


@patch("connoisseur.ingestion.data_manager.call_llm")
def test_new_data_entry_process_returns_record_on_first_try(mock_llm):
    mock_llm.return_value = VALID_RESTAURANT_JSON

    record = new_data_entry_process("Paragraph", 42)

    assert record["name"] == "Test Place"
    assert mock_llm.call_count == 1


# ---------------------------------------------------------------------------
# manage_restaurants — CLI menu paths
# ---------------------------------------------------------------------------

def _run_manage(inputs: list[str], initial_data=None, tmp_path=None):
    """Helper to run manage_restaurants with mocked input."""
    if initial_data is None:
        initial_data = [{"name": "Cafe Test", "location": "SF"}]
    file_path = str(tmp_path / "db.json")
    backup_path = file_path + ".bak"
    with open(file_path, "w") as f:
        json.dump(initial_data, f)

    with patch("builtins.input", side_effect=inputs):
        manage_restaurants(file_path, backup_path)

    return file_path, backup_path


def test_manage_exits_on_choice_6(tmp_path, capsys):
    _run_manage(["6"], tmp_path=tmp_path)
    # No exception means the loop exited cleanly


def test_manage_browse_lists_names(tmp_path, capsys):
    _run_manage(["1", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Cafe Test" in captured.out


def test_manage_invalid_choice_prints_error(tmp_path, capsys):
    _run_manage(["X", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out


def test_manage_view_detail_valid_index(tmp_path, capsys):
    _run_manage(["2", "0", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Cafe Test" in captured.out


def test_manage_view_detail_invalid_index(tmp_path, capsys):
    _run_manage(["2", "999", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Invalid index" in captured.out


def test_manage_view_detail_non_numeric_index(tmp_path, capsys):
    _run_manage(["2", "abc", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Invalid index" in captured.out


def test_manage_security_cancel_on_no(tmp_path, capsys):
    _run_manage(["3", "no", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Operation cancelled" in captured.out


def test_manage_security_cancel_on_empty(tmp_path, capsys):
    _run_manage(["5", "", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Operation cancelled" in captured.out


@patch("connoisseur.ingestion.data_manager.new_data_entry_process")
def test_manage_add_restaurant(mock_process, tmp_path, capsys):
    new_record = {"name": "New Place", "location": "LA"}
    mock_process.return_value = new_record

    file_path, _ = _run_manage(
        ["3", "yes", "Great new bistro description.", "6"],
        initial_data=[],
        tmp_path=tmp_path,
    )

    captured = capsys.readouterr()
    assert "added" in captured.out
    saved = json.loads(open(file_path).read())
    assert len(saved) == 1


def test_manage_delete_valid_index(tmp_path, capsys):
    file_path, _ = _run_manage(["5", "yes", "0", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Deleted" in captured.out
    saved = json.loads(open(file_path).read())
    assert saved == []


def test_manage_delete_invalid_index(tmp_path, capsys):
    _run_manage(["5", "yes", "999", "6"], tmp_path=tmp_path)
    captured = capsys.readouterr()
    assert "Invalid index" in captured.out


def test_manage_edit_skips_empty_input(tmp_path, capsys):
    # Initial record has 2 keys (name, location) → 2 skip inputs needed
    file_path, _ = _run_manage(
        ["4", "yes", "0", "", "", "6"],
        tmp_path=tmp_path,
    )
    saved = json.loads(open(file_path).read())
    assert saved[0]["name"] == "Cafe Test"  # unchanged
