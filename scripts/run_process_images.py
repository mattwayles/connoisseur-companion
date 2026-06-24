#!/usr/bin/env python3
"""M1L2 — Caption recipe images and user review images."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connoisseur.ingestion.process_images import caption_recipe_images, caption_review_images

if __name__ == "__main__":
    print("=== M1L2: Captioning recipe images ===")
    recipes = caption_recipe_images(verbose=True)
    print(f"Done. {len(recipes)} recipes augmented.\n")

    print("=== M1L2: Captioning user review images ===")
    reviews = caption_review_images(verbose=True)
    print(f"Done. {len(reviews)} reviews augmented.")
