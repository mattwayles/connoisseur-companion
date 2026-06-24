"""M4L1 — FastMCP server exposing California restaurant data via 1 resource and 3 tools."""
import json
from pathlib import Path

from fastmcp import FastMCP

# ── Data paths (relative to this file) ───────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
CULINARY_MAP_PATH    = _DATA_DIR / "California-Culinary-Map.txt"
RESTAURANT_DATA_PATH = _DATA_DIR / "structured_restaurant_data.json"
REVIEW_DATA_PATH     = _DATA_DIR / "augmented_user_review.json"

# ── Server instance ───────────────────────────────────────────────────────────
mcp = FastMCP("Connoisseur-Server")


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_restaurants() -> list[dict]:
    with open(RESTAURANT_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_reviews() -> list[dict]:
    with open(REVIEW_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Resource ──────────────────────────────────────────────────────────────────

@mcp.resource("culinary-map://california")
def get_culinary_map() -> str:
    """The full raw California Culinary Map text from Module 1.
    Contains detailed descriptions of 100+ restaurants across California
    including their vibes, cuisines, ratings, and price ranges."""
    return CULINARY_MAP_PATH.read_text(encoding="utf-8")


# ── Tool 1: exact / partial name lookup ──────────────────────────────────────

@mcp.tool()
def get_restaurant_info(restaurant_name: str) -> str:
    """Search for a restaurant by name and return its structured details
    including cuisine, rating, price range, vibe, and signature dishes.
    Supports partial matching — try 'Iron' to find 'Iron Chef Kitchen'."""
    restaurants = _load_restaurants()
    query = restaurant_name.lower().strip()

    matches = [
        r for r in restaurants
        if query in r.get("name", "").lower() or r.get("name", "").lower() in query
    ]

    if not matches:
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No restaurant found matching '{restaurant_name}'.",
                "suggestion": "Try a partial name like 'Iron', 'Sakura', or 'Gilded'.",
            },
            indent=2,
        )

    return json.dumps(
        {"status": "found", "count": len(matches), "results": matches},
        indent=2,
    )


# ── Tool 2: vibe / atmosphere search ─────────────────────────────────────────

@mcp.tool()
def recommend_by_vibe(vibe: str) -> str:
    """Find restaurants that match a given vibe or atmosphere keyword.
    Performs a two-pass search: structured vibe/environment fields first,
    then raw paragraph text from the culinary map.
    Example keywords: 'moody', 'sun-drenched', 'romantic', 'zen', 'cozy'."""
    restaurants = _load_restaurants()
    vibe_lower = vibe.lower().strip()

    # Pass 1 — structured fields
    structured_matches: list[dict] = []
    for r in restaurants:
        vibe_field = str(r.get("vibe", "")).lower()
        env_field  = str(r.get("environment", "")).lower()
        if vibe_lower in vibe_field or vibe_lower in env_field:
            structured_matches.append(
                {
                    "name":        r.get("name"),
                    "location":    r.get("location"),
                    "food_style":  r.get("food_style"),
                    "rating":      r.get("rating"),
                    "vibe":        r.get("vibe"),
                    "price_range": r.get("price_range"),
                }
            )

    # Pass 2 — raw paragraph text
    raw_text = CULINARY_MAP_PATH.read_text(encoding="utf-8")
    text_excerpts = [
        para.strip()[:300]
        for para in raw_text.split("\n\n")
        if vibe_lower in para.lower() and para.strip()
    ][:5]

    return json.dumps(
        {
            "vibe_searched":    vibe,
            "structured_matches": structured_matches,
            "raw_text_excerpts":  text_excerpts,
        },
        indent=2,
    )


# ── Tool 3: review retrieval ──────────────────────────────────────────────────

@mcp.tool()
def get_review(restaurant_name: str) -> str:
    """Retrieve the full user review for a restaurant.
    Searches by cross-referencing the restaurant name with itemId lookup,
    review title, and review text."""
    reviews     = _load_reviews()
    restaurants = _load_restaurants()
    query       = restaurant_name.lower().strip()

    # Build itemId → restaurant name map
    id_to_name: dict[int, str] = {
        r.get("itemId"): r.get("name", "")
        for r in restaurants
        if r.get("itemId")
    }

    matching: dict | None = None
    for review in reviews:
        rest_name  = id_to_name.get(review.get("itemId"), "")
        title_text = review.get("title", "")
        body_text  = review.get("text", "")
        if (
            query in rest_name.lower()
            or query in title_text.lower()
            or query in body_text.lower()
        ):
            matching = dict(review)
            matching["restaurant_name"] = rest_name
            break

    if not matching:
        return json.dumps(
            {
                "status":  "not_found",
                "message": f"No review found for '{restaurant_name}'.",
            },
            indent=2,
        )

    return json.dumps(
        {
            "status":           "found",
            "restaurant":       matching.get("restaurant_name", "Unknown"),
            "reviewer":         matching.get("userId", "Anonymous"),
            "rating":           matching.get("rating"),
            "title":            matching.get("title", ""),
            "review_text":      matching.get("text", ""),
            "image_captions":   matching.get("image_captions", []),
            "visit_date":       matching.get("date", "N/A"),
        },
        indent=2,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
