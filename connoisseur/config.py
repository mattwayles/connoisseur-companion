"""Central configuration: paths, model IDs, ChromaDB location."""
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent          # project/
DATA_DIR    = PROJECT_DIR / "data"                  # project/data/
IMAGE_DIR   = DATA_DIR / "synthetic_recipe_images"  # project/data/synthetic_recipe_images/
CHROMA_DIR  = Path.home() / "chroma_connoisseur"    # ~/chroma_connoisseur/

# ── Raw input files ───────────────────────────────────────────────────────────
CULINARY_MAP_PATH  = DATA_DIR / "California-Culinary-Map.txt"
RAW_RECIPES_PATH   = DATA_DIR / "Recipes.json"
RAW_REVIEWS_PATH   = DATA_DIR / "Synthetic-User-Reviews.json"

# ── Generated output files ────────────────────────────────────────────────────
RESTAURANT_DATA_PATH = DATA_DIR / "structured_restaurant_data.json"
RECIPE_DATA_PATH     = DATA_DIR / "augmented_food_recipe.json"
REVIEW_DATA_PATH     = DATA_DIR / "augmented_user_review.json"

# ── Anthropic model IDs ───────────────────────────────────────────────────────
CLAUDE_HAIKU  = "claude-haiku-4-5-20251001"
CLAUDE_SONNET = "claude-sonnet-4-6"

# ── ChromaDB collection names ─────────────────────────────────────────────────
ARTICLE_COLLECTION = "restaurant_articles"
IMAGE_COLLECTION   = "food_images"
