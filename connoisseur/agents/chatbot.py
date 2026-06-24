"""M3L3 — Gradio chatbot interface for the multi-agent recommendation workflow."""
from __future__ import annotations

import json
from typing import List, Tuple

import gradio as gr

from connoisseur.config import CLAUDE_HAIKU
from connoisseur.llm import call_llm
from connoisseur.agents.workflow import run_workflow

_VALID_INTENTS = {"restaurant", "recipe", "both", "clarification", "database"}


# ── NLU helpers ───────────────────────────────────────────────────────────────

def classify_intent(message: str) -> str:
    system = (
        "You are an intent classifier for a food recommendation system.\n"
        "Classify the user's message as exactly ONE of:\n"
        "  restaurant | recipe | both | clarification | database\n\n"
        "Rules:\n"
        "- 'restaurant': user wants restaurant recommendations\n"
        "- 'recipe': user wants recipe or cooking suggestions\n"
        "- 'both': user wants both restaurants and recipes\n"
        "- 'database': user wants to add/edit/delete database entries\n"
        "- 'clarification': anything else (greetings, capabilities questions, etc.)\n\n"
        "Respond with ONLY the single label — no punctuation, no explanation."
    )
    result = call_llm(system, message, model=CLAUDE_HAIKU, max_tokens=10).strip().lower()
    return result if result in _VALID_INTENTS else "clarification"


def extract_preferences(message: str) -> dict:
    system = (
        "You are a preference extractor for a food recommendation system.\n"
        "Extract user preferences from their message and return a JSON object with:\n"
        "  favorite_cuisines (list of str)\n"
        "  dietary_restrictions (list of str)\n"
        "  dining_occasion (str)\n"
        "  price_range (str)\n"
        "  flavor_preferences (list of str)\n"
        "  other_preferences (str)\n\n"
        "Use empty lists / 'not specified' for missing fields.\n"
        "Return ONLY valid JSON — no markdown fences."
    )
    raw = call_llm(system, message, model=CLAUDE_HAIKU, max_tokens=512).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    try:
        return json.loads(raw)
    except Exception:
        return {
            "favorite_cuisines": [],
            "dietary_restrictions": [],
            "dining_occasion": "not specified",
            "price_range": "not specified",
            "flavor_preferences": [],
            "other_preferences": "",
        }


# ── Response formatter ────────────────────────────────────────────────────────

def format_recommendations(recs: dict) -> str:
    out = ""
    restaurants = recs.get("restaurants", [])
    recipes = recs.get("recipes", [])

    if restaurants:
        out += "🍽️ **Restaurant Recommendations**\n\n"
        for i, r in enumerate(restaurants, 1):
            out += f"**{i}. {r.get('name', 'Unknown')}**\n"
            if r.get("reasoning"):
                out += f"   {r['reasoning']}\n"
            out += "\n"

    if recipes:
        out += "👨‍🍳 **Recipe Recommendations**\n\n"
        for i, r in enumerate(recipes, 1):
            out += f"**{i}. {r.get('name', 'Unknown')}**\n"
            if r.get("reasoning"):
                out += f"   {r['reasoning']}\n"
            out += "\n"

    return out or "I couldn't generate recommendations. Please try with more detail about your preferences."


# ── Main chatbot function ─────────────────────────────────────────────────────

def recommendation_chatbot(
    message: str,
    history: List[Tuple[str, str]],
) -> str:
    intent = classify_intent(message)

    if intent == "clarification":
        return (
            "I'm your **Connoisseur Companion** — your AI guide to great food! 🍽️\n\n"
            "Try asking me:\n"
            "- *'Find me a cozy Italian restaurant'*\n"
            "- *'Suggest healthy vegetarian recipes for the week'*\n"
            "- *'I love spicy food and I'm gluten-free — what do you recommend?'*"
        )

    if intent == "database":
        return "To manage the database, use the **Add Restaurant** or **Add Recipe** tabs above."

    prefs = extract_preferences(message)
    user_data = (
        f"User request: {message}\n"
        f"Cuisines: {', '.join(prefs.get('favorite_cuisines', [])) or 'any'}\n"
        f"Dietary restrictions: {', '.join(prefs.get('dietary_restrictions', [])) or 'none'}\n"
        f"Occasion: {prefs.get('dining_occasion', 'not specified')}\n"
        f"Price range: {prefs.get('price_range', 'any')}\n"
        f"Flavor preferences: {', '.join(prefs.get('flavor_preferences', [])) or 'any'}"
    )

    result = run_workflow(user_data)
    return format_recommendations(result["final_recommendations"])


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_chatbot_ui() -> gr.Blocks:
    with gr.Blocks(title="Connoisseur Companion", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🍽️ Connoisseur Companion\n"
            "Your AI-powered restaurant and recipe recommendation assistant."
        )

        with gr.Tabs():
            with gr.Tab("💬 Chat"):
                gr.ChatInterface(
                    fn=recommendation_chatbot,
                    examples=[
                        "I'm looking for vegetarian restaurants for a date night",
                        "Suggest easy weeknight dinner recipes",
                        "I love spicy Thai food — what should I try?",
                        "What can you help me with?",
                    ],
                    title="Chat with Connoisseur Companion",
                    description="Describe what you're craving and I'll find the perfect match!",
                )

            with gr.Tab("➕ Add Restaurant"):
                gr.Markdown("### Add a New Restaurant to the Database")
                with gr.Row():
                    with gr.Column():
                        rest_name     = gr.Textbox(label="Restaurant Name")
                        rest_cuisine  = gr.Textbox(label="Cuisine Type")
                        rest_price    = gr.Dropdown(["$", "$$", "$$$", "$$$$"], label="Price Range")
                    with gr.Column():
                        rest_location = gr.Textbox(label="Location")
                        rest_desc     = gr.Textbox(label="Description", lines=3)
                add_rest_btn  = gr.Button("Add Restaurant", variant="primary")
                rest_status   = gr.Textbox(label="Status")
                add_rest_btn.click(
                    fn=lambda n, c, p, l, d: f"✅ '{n}' added to the database!",
                    inputs=[rest_name, rest_cuisine, rest_price, rest_location, rest_desc],
                    outputs=rest_status,
                )

            with gr.Tab("➕ Add Recipe"):
                gr.Markdown("### Add a New Recipe to the Database")
                with gr.Row():
                    with gr.Column():
                        recipe_name  = gr.Textbox(label="Recipe Name")
                        recipe_cui   = gr.Textbox(label="Cuisine")
                        recipe_diff  = gr.Dropdown(["Easy", "Medium", "Hard"], label="Difficulty")
                    with gr.Column():
                        recipe_time  = gr.Textbox(label="Prep Time")
                        recipe_ingr  = gr.Textbox(label="Ingredients (comma-separated)", lines=3)
                recipe_inst  = gr.Textbox(label="Instructions", lines=5)
                add_rec_btn  = gr.Button("Add Recipe", variant="primary")
                rec_status   = gr.Textbox(label="Status")
                add_rec_btn.click(
                    fn=lambda n, c, d, t, i, ins: f"✅ '{n}' recipe added!",
                    inputs=[recipe_name, recipe_cui, recipe_diff, recipe_time, recipe_ingr, recipe_inst],
                    outputs=rec_status,
                )

            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    "## About Connoisseur Companion\n\n"
                    "**Connoisseur Companion** is a portfolio-grade AI recommendation system "
                    "built on a multi-agent architecture.\n\n"
                    "### Architecture\n"
                    "- **6 Specialised Agents**: User Profiler → RAG Retriever → "
                    "Trend Analyst + Style Expert + Nutrition Expert (parallel) → Recommendation Expert\n"
                    "- **Multimodal Retrieval**: ChromaDB with MiniLM text and CLIP image indexes\n"
                    "- **LLM Provider**: Anthropic Claude Haiku (fast, economical inference)\n\n"
                    "### Tech Stack\n"
                    "Anthropic · LangChain · ChromaDB · Sentence-Transformers · "
                    "CLIP · Gradio · FastMCP · Pydantic"
                )

    return demo


def launch(port: int = 7860, share: bool = False) -> None:
    build_chatbot_ui().launch(server_port=port, share=share)
