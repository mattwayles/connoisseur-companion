"""M3L2 — Four-phase multi-agent workflow with parallel Phase 3 execution."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from connoisseur.config import CLAUDE_HAIKU
from connoisseur.llm import call_llm
from connoisseur.agents.configs import AGENT_CONFIGS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_system(agent_key: str) -> str:
    cfg = AGENT_CONFIGS[agent_key]
    return (
        f"You are a {cfg['role']}.\n\n"
        f"Your goal: {cfg['goal']}\n\n"
        f"Your background: {cfg['backstory']}\n\n"
        "Respond with structured, actionable JSON output as specified."
    )


def _call_agent(agent_key: str, message: str) -> str:
    return call_llm(_make_system(agent_key), message, model=CLAUDE_HAIKU, max_tokens=4096)


def _parse_json(text: str) -> dict | list:
    """Strip optional markdown fences then parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


# ── Phase nodes ───────────────────────────────────────────────────────────────

def node_generate_profile(state: dict) -> dict:
    print("[Phase 1] Generating user profile...")
    msg = (
        f"Analyze this user data and create a comprehensive dining profile:\n\n{state['user_input']}\n\n"
        "Return JSON with keys: favorite_cuisines (list), dietary_restrictions (list), "
        "dining_occasions (list), price_range (str), adventurousness_score (int 1-10), "
        "flavor_preferences (list), summary (str)."
    )
    try:
        result = _parse_json(_call_agent("user_profile_generator", msg))
        print(f"  ✓ {result.get('summary', 'Profile generated')[:80]}")
    except Exception as exc:
        result = {"error": str(exc)}

    state["user_profile"] = result
    state["workflow_step"] = "profile_generated"
    return state


def node_retrieve_candidates(state: dict) -> dict:
    print("[Phase 2] Retrieving candidates...")
    msg = (
        f"Based on this user profile, generate diverse restaurant and recipe candidates:\n"
        f"{json.dumps(state['user_profile'], indent=2)}\n\n"
        "Return JSON with:\n"
        "- restaurants: [{name, cuisine, price, rating, description}] (20 items)\n"
        "- recipes: [{name, cuisine, difficulty, prep_time, description}] (20 items)"
    )
    try:
        result = _parse_json(_call_agent("rag_retriever", msg))
        restaurants = result.get("restaurants", [])
        recipes = result.get("recipes", [])
        print(f"  ✓ {len(restaurants)} restaurants, {len(recipes)} recipes")
    except Exception as exc:
        restaurants, recipes = [], []
        print(f"  ⚠ {exc}")

    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state


def node_analyze_trends(state: dict) -> dict:
    print("[Phase 3a] Analyzing food trends...")
    msg = (
        f"Identify 3-5 current food trends relevant to these candidates:\n"
        f"Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}\n"
        f"Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}\n\n"
        'Return JSON: {"trends": [{"name": str, "description": str, "relevance": str}]}'
    )
    try:
        result = _parse_json(_call_agent("food_trend_analyst", msg))
        print(f"  ✓ {len(result.get('trends', []))} trends identified")
    except Exception as exc:
        result = {"trends": [], "error": str(exc)}

    state["trend_analysis"] = result
    return state


def node_analyze_styles(state: dict) -> dict:
    print("[Phase 3b] Analyzing food styles...")
    msg = (
        f"Analyze the cuisine types and flavor profiles in these candidates:\n"
        f"Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}\n"
        f"Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}\n\n"
        'Return JSON: {"cuisines": [{"name": str, "description": str}], '
        '"profiles": [{"name": str, "description": str}]}'
    )
    try:
        result = _parse_json(_call_agent("food_style_expert", msg))
        print("  ✓ Style analysis complete")
    except Exception as exc:
        result = {"cuisines": [], "profiles": [], "error": str(exc)}

    state["style_analysis"] = result
    return state


def node_evaluate_nutrition(state: dict) -> dict:
    print("[Phase 3c] Evaluating nutrition...")
    msg = (
        f"Evaluate nutritional fit for this user:\n"
        f"Profile: {json.dumps(state['user_profile'], indent=2)}\n"
        f"Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}\n"
        f"Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}\n\n"
        'Return JSON: {"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}'
    )
    try:
        result = _parse_json(_call_agent("nutrition_expert", msg))
        print("  ✓ Nutrition evaluation complete")
    except Exception as exc:
        result = {"compliant_items": [], "flagged_items": [], "nutritional_highlights": [], "error": str(exc)}

    state["nutrition_analysis"] = result
    return state


def node_generate_recommendations(state: dict) -> dict:
    print("[Phase 4] Generating final recommendations...")
    msg = (
        f"Synthesize all insights into top-5 restaurant and top-5 recipe recommendations:\n\n"
        f"User Profile: {json.dumps(state['user_profile'], indent=2)}\n"
        f"Restaurants ({len(state['retrieved_restaurants'])} candidates): "
        f"{json.dumps(state['retrieved_restaurants'][:10], indent=2)}\n"
        f"Recipes ({len(state['retrieved_recipes'])} candidates): "
        f"{json.dumps(state['retrieved_recipes'][:10], indent=2)}\n"
        f"Trends: {json.dumps(state['trend_analysis'], indent=2)}\n"
        f"Styles: {json.dumps(state['style_analysis'], indent=2)}\n"
        f"Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}\n\n"
        "Return JSON:\n"
        '{"restaurants": [{"name": str, "reasoning": str}], '
        '"recipes": [{"name": str, "reasoning": str}]}\n'
        "Each reasoning should be 2-3 sentences explaining the personal match."
    )
    try:
        result = _parse_json(_call_agent("recommendation_expert", msg))
        print(
            f"  ✓ {len(result.get('restaurants', []))} restaurant recs, "
            f"{len(result.get('recipes', []))} recipe recs"
        )
    except Exception as exc:
        result = {"restaurants": [], "recipes": [], "error": str(exc)}

    state["final_recommendations"] = result
    state["workflow_step"] = "complete"
    return state


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_workflow(user_input: str) -> dict:
    """Execute the full four-phase multi-agent workflow.

    Phases:
        1. User Analysis    (sequential)
        2. Data Retrieval   (sequential)
        3. Analysis         (parallel — trends, styles, nutrition)
        4. Synthesis        (sequential)
    """
    state: dict = {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start",
    }

    # Phase 1 & 2 — sequential
    state = node_generate_profile(state)
    state = node_retrieve_candidates(state)

    # Phase 3 — parallel
    print("\n[Phase 3] Running analysis agents in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_trends    = executor.submit(node_analyze_trends,    dict(state))
        f_styles    = executor.submit(node_analyze_styles,    dict(state))
        f_nutrition = executor.submit(node_evaluate_nutrition, dict(state))

        state["trend_analysis"]    = f_trends.result()["trend_analysis"]
        state["style_analysis"]    = f_styles.result()["style_analysis"]
        state["nutrition_analysis"] = f_nutrition.result()["nutrition_analysis"]

    # Phase 4 — sequential
    state = node_generate_recommendations(state)
    return state
