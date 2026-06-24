"""M1L2 — Caption food images and augment recipe / user-review JSON records."""
import ast
import base64
import json
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from connoisseur.config import (
    DATA_DIR,
    IMAGE_DIR,
    RAW_RECIPES_PATH,
    RAW_REVIEWS_PATH,
    RECIPE_DATA_PATH,
    REVIEW_DATA_PATH,
)
from connoisseur.llm import call_vision_llm


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _b64_from_path(path: Path) -> tuple[str, str]:
    """Base64-encode a local image file; return (b64_data, media_type)."""
    suffix = path.suffix.lower().lstrip(".")
    media_type = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return base64.b64encode(path.read_bytes()).decode(), media_type


def _b64_from_bytes(data: bytes, media_type: str = "image/jpeg") -> tuple[str, str]:
    return base64.b64encode(data).decode(), media_type


# ── Prompt builders ───────────────────────────────────────────────────────────

def _recipe_caption_prompt(food_name: str) -> tuple[str, str]:
    system = (
        "You are an objective culinary AI assistant. Describe the provided recipe image with "
        "absolute accuracy. Focus exclusively on observable visual facts: visible ingredients, "
        "cooking style, textures, and plating. Be concise and factual."
    )
    prompt = (
        f"This is an image of '{food_name}'. Describe the visible ingredients, "
        "the evident cooking style (e.g., grilled, roasted, raw), and how the dish is presented."
    )
    return system, prompt


def _review_caption_prompt(review_text: str) -> tuple[str, str]:
    system = (
        "You are a sophisticated culinary expert. Analyze a food image objectively while "
        "cross-referencing provided customer review text. Be concise, factual, and aligned "
        "with what is directly visible in the image."
    )
    prompt = (
        f"Customer review context:\n{review_text}\n\n"
        "Describe the food image in light of this review, focusing on relevant visual elements "
        "(plating, cooking style, ingredients) that match or reflect the reviewer's experience."
    )
    return system, prompt


# ── Network helper ────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_url(url: str) -> bytes:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.content


# ── Public pipeline functions ─────────────────────────────────────────────────

def caption_recipe_images(verbose: bool = True) -> list[dict]:
    """Generate image captions for every recipe and save augmented_food_recipe.json."""
    with open(RAW_RECIPES_PATH, encoding="utf-8") as f:
        recipe_data: list[dict] = json.load(f)

    for i, recipe in enumerate(recipe_data):
        img_path = IMAGE_DIR / f"recipe{i + 1}.png"
        if not img_path.exists():
            recipe["image_description"] = ""
            continue

        b64, media_type = _b64_from_path(img_path)
        system, prompt = _recipe_caption_prompt(recipe.get("name", f"recipe {i + 1}"))
        recipe["image_description"] = call_vision_llm(system, prompt, b64, media_type)

        if verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(recipe_data)} recipes captioned")

    RECIPE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RECIPE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(recipe_data, f, indent=4)

    if verbose:
        print(f"✅ Saved augmented recipes → {RECIPE_DATA_PATH}")

    return recipe_data


def caption_review_images(verbose: bool = True) -> list[dict]:
    """Fetch and caption images from user reviews; save augmented_user_review.json."""
    with open(RAW_REVIEWS_PATH, encoding="utf-8") as f:
        review_data: list[dict] = json.load(f)

    placeholder = DATA_DIR / "_review_image_tmp.jpg"

    for i, review in enumerate(review_data):
        raw_images = review.get("images", "[]")
        image_urls: list[str] = ast.literal_eval(raw_images) if isinstance(raw_images, str) else raw_images
        captions: list[str] = []

        for url in image_urls:
            try:
                img_bytes = _fetch_url(url)
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ Failed to fetch {url}: {exc}")
                continue

            # Persist temporarily so we can detect format
            placeholder.write_bytes(img_bytes)
            b64, media_type = _b64_from_bytes(img_bytes, "image/jpeg")
            system, prompt = _review_caption_prompt(review.get("text", ""))
            captions.append(call_vision_llm(system, prompt, b64, media_type))

        review["image_captions"] = captions

    if placeholder.exists():
        placeholder.unlink()

    REVIEW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=4)

    if verbose:
        print(f"✅ Saved augmented reviews → {REVIEW_DATA_PATH}")

    return review_data
