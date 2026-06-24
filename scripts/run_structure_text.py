#!/usr/bin/env python3
"""M1L1 — Structure all restaurant text paragraphs into JSON records."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connoisseur.ingestion.structure_text import run_structuring

if __name__ == "__main__":
    print("=== M1L1: Structuring restaurant text ===")
    records = run_structuring(verbose=True)
    print(f"\nDone. {len(records)} restaurants structured.")
