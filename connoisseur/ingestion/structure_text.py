"""M1L1 — Convert unstructured restaurant paragraphs into validated JSON records."""
import json
import time
from pydantic import ValidationError

from connoisseur.config import CULINARY_MAP_PATH, RESTAURANT_DATA_PATH
from connoisseur.llm import call_llm
from connoisseur.ingestion.models import Restaurant

# ── One-shot example used in every structuring prompt ────────────────────────
_EXAMPLE_PARAGRAPH = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, "
    "**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, "
    "it captures the salt-air energy of the coast through its signature beer-battered snapper "
    "tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. "
    "Price range: $"
)

_EXAMPLE_OUTPUT = """{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": ["beer-battered snapper tacos", "zesty octopus ceviche"],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def restaurant_data_structure_prompt(paragraph: str) -> tuple[str, str]:
    """Return (system, user) prompt pair for structuring a restaurant paragraph."""
    system = (
        "You are a helpful assistant skilled in converting unstructured text into structured JSON output."
    )
    prompt = f"""Task: Convert the restaurant description below into a JSON object matching the example.
For price_range, convert dollar signs ($, $$, $$$, $$$$) to the integer count of dollar symbols.

Restaurant description:
{paragraph}

Example input:
{_EXAMPLE_PARAGRAPH}

Example output:
{_EXAMPLE_OUTPUT}

Return ONLY the raw JSON object — no markdown fences, no explanation."""
    return system, prompt


def json_auto_repair_prompt(candidate: str, error: str) -> tuple[str, str]:
    """Return (system, user) prompt pair to repair malformed JSON."""
    system = (
        "You are a strict JSON repair assistant. "
        "Fix syntax and structural errors based on the validation message. "
        "Return ONLY the corrected raw JSON — no markdown, no explanation."
    )
    prompt = f"""Fix the invalid JSON below so it resolves the validation error.

Invalid JSON:
{candidate}

Validation error:
{error}

Corrected JSON:"""
    return system, prompt


# ── Core pipeline ─────────────────────────────────────────────────────────────

def load_raw_paragraphs() -> list[str]:
    """Read the culinary map and split into per-restaurant paragraphs."""
    text = CULINARY_MAP_PATH.read_text(encoding="utf-8")
    parts = text.split("\n\n")
    # First item is the dataset header line — skip it
    return [p.strip() for p in parts[1:] if p.strip()]


def structure_paragraph(paragraph: str) -> str:
    """Run one paragraph through the LLM and return a validated JSON string.

    Retries up to three times using the auto-repair agent on validation failure.
    """
    system, prompt = restaurant_data_structure_prompt(paragraph)
    result = call_llm(system, prompt)

    for _ in range(3):
        try:
            Restaurant.model_validate_json(result)
            return result
        except (ValidationError, Exception) as exc:
            repair_sys, repair_prompt = json_auto_repair_prompt(result, str(exc))
            result = call_llm(repair_sys, repair_prompt)
            time.sleep(0.1)

    return result  # best-effort — caller decides how to handle residual errors


def run_structuring(verbose: bool = True) -> list[dict]:
    """Structure every restaurant paragraph and save to disk.

    Returns the list of structured restaurant dicts.
    """
    paragraphs = load_raw_paragraphs()
    structured: list[dict] = []

    for i, paragraph in enumerate(paragraphs):
        raw = structure_paragraph(paragraph)
        try:
            record = json.loads(raw)
        except Exception:
            record = {
                "name": f"Unknown_{i}",
                "location": "",
                "type": "",
                "food_style": "",
                "environment": "",
            }

        record["itemId"] = 1000001 + i
        structured.append(record)

        if verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(paragraphs)} done")

    RESTAURANT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESTAURANT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=4)

    if verbose:
        print(f"✅ Saved {len(structured)} records → {RESTAURANT_DATA_PATH}")

    return structured
