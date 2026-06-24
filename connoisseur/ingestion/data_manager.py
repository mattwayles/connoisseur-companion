"""M1L3 — Interactive CLI for CRUD operations on the structured restaurant database."""
import io
import json
import os
import shutil
import time
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from connoisseur.config import RESTAURANT_DATA_PATH
from connoisseur.ingestion.models import Restaurant
from connoisseur.ingestion.structure_text import (
    restaurant_data_structure_prompt,
    json_auto_repair_prompt,
)
from connoisseur.llm import call_llm

FILEPATH = str(RESTAURANT_DATA_PATH)
BACKUP_PATH = FILEPATH + ".bak"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(data: list[dict], file_path: str, backup_path: str) -> None:
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(res: dict, index: int) -> None:
    print(f"\n{'='*15} RESTAURANT #{index} {'='*15}")
    name = res.get("name", res.get("restaurant_name", "Unnamed Restaurant"))
    print(f"NAME        : {name}")
    for key, value in res.items():
        if key.lower() not in ("name", "restaurant_name"):
            label = key.replace("_", " ").upper()
            print(f"{label:<12}: {value}")
    print("=" * 45)


# ── LLM-powered new-entry processing ─────────────────────────────────────────

def new_data_entry_process(paragraph: str, item_id: int) -> dict:
    """Structure a freeform restaurant description and assign it the given itemId."""
    system, prompt = restaurant_data_structure_prompt(paragraph)
    result = call_llm(system, prompt)

    for _ in range(3):
        try:
            Restaurant.model_validate_json(result)
            break
        except (ValidationError, Exception) as exc:
            repair_sys, repair_prompt = json_auto_repair_prompt(result, str(exc))
            result = call_llm(repair_sys, repair_prompt)
            time.sleep(0.1)

    record = json.loads(result)
    record["itemId"] = item_id
    return record


# ── Main UI ───────────────────────────────────────────────────────────────────

def manage_restaurants(
    file_path: str = FILEPATH,
    backup_path: str = BACKUP_PATH,
) -> None:
    """Interactive command-line UI for managing the restaurant database."""
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ").strip()

        if choice == "1":
            print("\n--- Current Listings ---")
            for idx, r in enumerate(data):
                print(f"  [{idx}] {r.get('name', 'N/A')}")

        elif choice == "2":
            try:
                idx = int(input("Enter record index: "))
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid index.")

        elif choice in ("3", "4", "5"):
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").strip().lower()
            if confirm != "yes":
                print("Operation cancelled.")
                continue

            if choice == "3":
                item_id = 1000000 + len(data) + 1
                paragraph = input("Enter new restaurant description:\n> ").strip()
                record = new_data_entry_process(paragraph, item_id)
                data.append(record)
                save_data(data, file_path, backup_path)
                print("✅ Restaurant added.")

            elif choice == "4":
                try:
                    idx = int(input("Enter record index to edit: "))
                    if 0 <= idx < len(data):
                        record = data[idx]
                        for key in list(record.keys()):
                            val = input(f"  {key} [{record[key]}] (Enter to skip): ").strip()
                            if val:
                                try:
                                    record[key] = json.loads(val)
                                except Exception:
                                    record[key] = val
                        data[idx] = record
                        save_data(data, file_path, backup_path)
                        print("✅ Record updated.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid index.")

            elif choice == "5":
                try:
                    idx = int(input("Enter record index to delete: "))
                    if 0 <= idx < len(data):
                        removed = data.pop(idx)
                        save_data(data, file_path, backup_path)
                        print(f"✅ Deleted: {removed.get('name', 'record')}")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid index.")

        elif choice == "6":
            break
        else:
            print("Invalid input.")


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        self.test_file = "structured_restaurant_data_unit_test.json"
        self.test_backup = self.test_file + ".bak"
        initial = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, "w") as f:
            json.dump(initial, f)

    def tearDown(self):
        for path in (self.test_file, self.test_backup):
            if os.path.exists(path):
                os.remove(path)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        mock_paragraph = (
            "The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination "
            "that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed "
            "wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$$ "
            "category, the menu celebrates seasonal foraging and local heritage, headlined by "
            "signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and "
            "hand-foraged Wild Mushroom Risotto with aged goat cheese."
        )
        mock_input.side_effect = ["3", "yes", mock_paragraph, "6"]
        try:
            manage_restaurants(self.test_file, self.test_backup)
        except SystemExit:
            pass

        with open(self.test_file) as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ["5", "yes", "1", "6"]
        try:
            manage_restaurants(self.test_file, self.test_backup)
        except SystemExit:
            pass

        with open(self.test_file) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        mock_input.side_effect = ["5", "no", "6"]
        manage_restaurants(self.test_file, self.test_backup)
        with open(self.test_file) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    # unittest.main()               # run unit tests
    manage_restaurants(FILEPATH, BACKUP_PATH)  # run interactive UI
