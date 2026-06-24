#!/usr/bin/env python3
"""M1L3 — Launch the interactive restaurant database CLI.

Pass --test to run unit tests instead of the interactive UI.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    if "--test" in sys.argv:
        from connoisseur.ingestion.data_manager import TestRestaurantDatabase
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRestaurantDatabase)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        from connoisseur.ingestion.data_manager import manage_restaurants, FILEPATH, BACKUP_PATH
        manage_restaurants(FILEPATH, BACKUP_PATH)
