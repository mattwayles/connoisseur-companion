#!/usr/bin/env python3
"""M2L1 — Build the multimodal ChromaDB vector index."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connoisseur.retrieval.build_index import build_index

if __name__ == "__main__":
    print("=== M2L1: Building multimodal vector index ===")
    article_db, image_db = build_index(reset=True)
    print(f"\nArticle vectors: {article_db._collection.count()}")
    print(f"Image vectors:   {image_db._collection.count()}")
