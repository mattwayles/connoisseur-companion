"""Tests for connoisseur.config — path layout and model identifiers."""
from pathlib import Path
from connoisseur import config


def test_project_dir_is_path():
    assert isinstance(config.PROJECT_DIR, Path)


def test_project_dir_is_absolute():
    assert config.PROJECT_DIR.is_absolute()


def test_data_dir_is_child_of_project():
    assert config.DATA_DIR == config.PROJECT_DIR / "data"


def test_image_dir_is_under_data():
    assert config.IMAGE_DIR.parent == config.DATA_DIR


def test_chroma_dir_is_under_home():
    assert config.CHROMA_DIR.parent == Path.home()
    assert "chroma_connoisseur" in config.CHROMA_DIR.name


def test_culinary_map_path_in_data():
    assert config.CULINARY_MAP_PATH.parent == config.DATA_DIR
    assert config.CULINARY_MAP_PATH.suffix == ".txt"


def test_raw_recipes_path_in_data():
    assert config.RAW_RECIPES_PATH.parent == config.DATA_DIR
    assert config.RAW_RECIPES_PATH.suffix == ".json"


def test_raw_reviews_path_in_data():
    assert config.RAW_REVIEWS_PATH.parent == config.DATA_DIR
    assert config.RAW_REVIEWS_PATH.suffix == ".json"


def test_generated_paths_in_data():
    for path in (
        config.RESTAURANT_DATA_PATH,
        config.RECIPE_DATA_PATH,
        config.REVIEW_DATA_PATH,
    ):
        assert path.parent == config.DATA_DIR
        assert path.suffix == ".json"


def test_model_ids_are_nonempty_strings():
    assert isinstance(config.CLAUDE_HAIKU, str) and config.CLAUDE_HAIKU
    assert isinstance(config.CLAUDE_SONNET, str) and config.CLAUDE_SONNET


def test_model_ids_are_distinct():
    assert config.CLAUDE_HAIKU != config.CLAUDE_SONNET


def test_collection_names_are_nonempty_strings():
    assert isinstance(config.ARTICLE_COLLECTION, str) and config.ARTICLE_COLLECTION
    assert isinstance(config.IMAGE_COLLECTION, str) and config.IMAGE_COLLECTION


def test_collection_names_are_distinct():
    assert config.ARTICLE_COLLECTION != config.IMAGE_COLLECTION
