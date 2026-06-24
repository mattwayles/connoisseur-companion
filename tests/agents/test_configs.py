"""Tests for connoisseur.agents.configs — agent configuration completeness."""
import pytest
from connoisseur.agents.configs import AGENT_CONFIGS, AGENT_TASKS

EXPECTED_AGENT_KEYS = {
    "user_profile_generator",
    "rag_retriever",
    "food_trend_analyst",
    "food_style_expert",
    "nutrition_expert",
    "recommendation_expert",
}

EXPECTED_TASK_KEYS = {
    "generate_profile",
    "retrieve_candidates",
    "analyze_trends",
    "analyze_styles",
    "evaluate_nutrition",
    "generate_recommendations",
}

CONFIG_REQUIRED_FIELDS = {"role", "goal", "backstory"}
TASK_REQUIRED_FIELDS = {"description", "expected_output", "agent"}


# ---------------------------------------------------------------------------
# AGENT_CONFIGS structure
# ---------------------------------------------------------------------------

def test_all_agent_keys_present():
    assert set(AGENT_CONFIGS.keys()) == EXPECTED_AGENT_KEYS


@pytest.mark.parametrize("key", sorted(EXPECTED_AGENT_KEYS))
def test_agent_config_has_required_fields(key):
    assert CONFIG_REQUIRED_FIELDS.issubset(set(AGENT_CONFIGS[key].keys()))


@pytest.mark.parametrize("key", sorted(EXPECTED_AGENT_KEYS))
def test_agent_config_fields_are_nonempty_strings(key):
    cfg = AGENT_CONFIGS[key]
    for field in CONFIG_REQUIRED_FIELDS:
        assert isinstance(cfg[field], str) and cfg[field].strip()


@pytest.mark.parametrize("key", sorted(EXPECTED_AGENT_KEYS))
def test_agent_role_is_title_case_words(key):
    role = AGENT_CONFIGS[key]["role"]
    assert len(role) > 0


# ---------------------------------------------------------------------------
# AGENT_TASKS structure
# ---------------------------------------------------------------------------

def test_all_task_keys_present():
    assert set(AGENT_TASKS.keys()) == EXPECTED_TASK_KEYS


@pytest.mark.parametrize("key", sorted(EXPECTED_TASK_KEYS))
def test_task_has_required_fields(key):
    assert TASK_REQUIRED_FIELDS.issubset(set(AGENT_TASKS[key].keys()))


@pytest.mark.parametrize("key", sorted(EXPECTED_TASK_KEYS))
def test_task_fields_are_nonempty_strings(key):
    task = AGENT_TASKS[key]
    for field in TASK_REQUIRED_FIELDS:
        assert isinstance(task[field], str) and task[field].strip()


@pytest.mark.parametrize("key", sorted(EXPECTED_TASK_KEYS))
def test_task_agent_references_valid_config(key):
    agent_ref = AGENT_TASKS[key]["agent"]
    assert agent_ref in AGENT_CONFIGS, f"Task '{key}' references unknown agent '{agent_ref}'"


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

def test_all_task_agents_are_used():
    """Every AGENT_CONFIGS key should appear in at least one task."""
    used_agents = {task["agent"] for task in AGENT_TASKS.values()}
    for agent_key in AGENT_CONFIGS:
        assert agent_key in used_agents, f"Agent '{agent_key}' has no associated task"


def test_configs_is_dict_of_dicts():
    assert isinstance(AGENT_CONFIGS, dict)
    for v in AGENT_CONFIGS.values():
        assert isinstance(v, dict)


def test_tasks_is_dict_of_dicts():
    assert isinstance(AGENT_TASKS, dict)
    for v in AGENT_TASKS.values():
        assert isinstance(v, dict)
