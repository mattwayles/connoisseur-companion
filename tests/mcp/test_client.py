"""Tests for connoisseur.mcp.client — roots, sampling callback, and tool invocation."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import connoisseur.mcp.client as client_module
from connoisseur.mcp.client import list_roots, handle_sampling


# ---------------------------------------------------------------------------
# list_roots
# ---------------------------------------------------------------------------

def test_list_roots_returns_list():
    roots = list_roots()
    assert isinstance(roots, list)
    assert len(roots) == 1


def test_list_roots_uri_is_file_scheme():
    roots = list_roots()
    assert roots[0].uri.startswith("file://")


def test_list_roots_uri_contains_project_dir():
    roots = list_roots()
    project_dir_str = str(client_module.PROJECT_DIR)
    assert project_dir_str in roots[0].uri


def test_list_roots_name_matches_dir_name():
    roots = list_roots()
    assert roots[0].name == client_module.PROJECT_DIR.name


# ---------------------------------------------------------------------------
# handle_sampling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_sampling_calls_anthropic():
    params = MagicMock()
    params.messages = [MagicMock(content=MagicMock(text="Tell me about sushi."))]
    params.maxTokens = 100

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Sushi is a Japanese dish.")]

    with patch.object(client_module._anthropic.messages, "create", return_value=mock_response):
        result = await handle_sampling(params)

    assert result.content.text == "Sushi is a Japanese dish."
    assert result.role == "assistant"


@pytest.mark.asyncio
async def test_handle_sampling_uses_max_tokens_from_params():
    params = MagicMock()
    params.messages = [MagicMock(content=MagicMock(text="prompt"))]
    params.maxTokens = 250

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response")]

    with patch.object(client_module._anthropic.messages, "create", return_value=mock_response) as mock_create:
        await handle_sampling(params)

    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["max_tokens"] == 250


@pytest.mark.asyncio
async def test_handle_sampling_defaults_max_tokens_when_none():
    params = MagicMock()
    params.messages = [MagicMock(content=MagicMock(text="prompt"))]
    params.maxTokens = None

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]

    with patch.object(client_module._anthropic.messages, "create", return_value=mock_response) as mock_create:
        await handle_sampling(params)

    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["max_tokens"] == 200  # falls back to `or 200`


@pytest.mark.asyncio
async def test_handle_sampling_result_model_is_haiku():
    params = MagicMock()
    params.messages = [MagicMock(content=MagicMock(text="hello"))]
    params.maxTokens = 50

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="hi")]

    with patch.object(client_module._anthropic.messages, "create", return_value=mock_response):
        result = await handle_sampling(params)

    assert "haiku" in result.model.lower()


# ---------------------------------------------------------------------------
# call_tool (async session mock)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_tool_returns_parsed_json():
    expected_data = {"status": "found", "count": 1}

    mock_content = MagicMock()
    mock_content.text = json.dumps(expected_data)
    mock_result = MagicMock()
    mock_result.content = [mock_content]

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_read_write = (AsyncMock(), AsyncMock())
    mock_stdio = AsyncMock()
    mock_stdio.__aenter__ = AsyncMock(return_value=mock_read_write)
    mock_stdio.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("connoisseur.mcp.client.stdio_client", return_value=mock_stdio),
        patch("connoisseur.mcp.client.ClientSession", return_value=mock_session),
    ):
        result = await client_module.call_tool("get_restaurant_info", {"restaurant_name": "Gilded"})

    assert result == expected_data
