"""Tests for connoisseur.mcp.app — Gradio MCP host application and ReAct loop."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import connoisseur.mcp.app as app_module
from connoisseur.mcp.app import chat_with_agent, handle_chat, build_app


# ---------------------------------------------------------------------------
# build_app
# ---------------------------------------------------------------------------

def test_build_app_returns_gradio_blocks():
    import gradio as gr
    demo = build_app()
    assert isinstance(demo, gr.Blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anthropic_response(stop_reason: str, text: str | None = None, tool_name: str | None = None):
    """Build a minimal mock Anthropic messages response."""
    response = MagicMock()
    response.stop_reason = stop_reason

    if stop_reason == "end_turn":
        block = MagicMock()
        block.text = text
        response.content = [block]
    elif stop_reason == "tool_use":
        block = MagicMock()
        block.type = "tool_use"
        block.name = tool_name
        block.input = {"restaurant_name": "Gilded"}
        block.id = "tool_use_123"
        response.content = [block]

    return response


# ---------------------------------------------------------------------------
# chat_with_agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_with_agent_end_turn_returns_text():
    end_response = _make_anthropic_response("end_turn", text="Here are some great restaurants!")

    mock_tool = MagicMock()
    mock_tool.name = "get_restaurant_info"
    mock_tool.description = "Look up a restaurant"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    mock_client = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[mock_tool])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("connoisseur.mcp.app.Client", return_value=mock_client),
        patch("connoisseur.mcp.app._anthropic.messages.create", return_value=end_response),
        patch("connoisseur.mcp.app.PythonStdioTransport"),
    ):
        result = await chat_with_agent("Find me a restaurant", [])

    assert result == "Here are some great restaurants!"


@pytest.mark.asyncio
async def test_chat_with_agent_tool_use_then_end_turn():
    """Agent calls a tool once, then produces a final text response."""
    tool_response = _make_anthropic_response("tool_use", tool_name="get_restaurant_info")
    end_response = _make_anthropic_response("end_turn", text="Found your restaurant!")

    mock_tool_obj = MagicMock()
    mock_tool_obj.name = "get_restaurant_info"
    mock_tool_obj.description = "Look up a restaurant"
    mock_tool_obj.inputSchema = {}

    tool_result_content = MagicMock()
    tool_result_content.text = '{"status": "found"}'
    mock_tool_result = MagicMock()
    mock_tool_result.content = [tool_result_content]

    mock_client = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[mock_tool_obj])
    mock_client.call_tool = AsyncMock(return_value=mock_tool_result)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("connoisseur.mcp.app.Client", return_value=mock_client),
        patch(
            "connoisseur.mcp.app._anthropic.messages.create",
            side_effect=[tool_response, end_response],
        ),
        patch("connoisseur.mcp.app.PythonStdioTransport"),
    ):
        result = await chat_with_agent("Tell me about Gilded", [])

    assert result == "Found your restaurant!"
    mock_client.call_tool.assert_called_once()


@pytest.mark.asyncio
async def test_chat_with_agent_with_existing_history():
    end_response = _make_anthropic_response("end_turn", text="Here you go!")

    mock_tool = MagicMock()
    mock_tool.name = "get_restaurant_info"
    mock_tool.description = "desc"
    mock_tool.inputSchema = {}

    mock_client = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[mock_tool])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    with (
        patch("connoisseur.mcp.app.Client", return_value=mock_client),
        patch("connoisseur.mcp.app._anthropic.messages.create", return_value=end_response) as mock_create,
        patch("connoisseur.mcp.app.PythonStdioTransport"),
    ):
        await chat_with_agent("Follow-up question", history)

    messages_sent = mock_create.call_args[1]["messages"]
    roles = [m["role"] for m in messages_sent if isinstance(m.get("content"), str)]
    assert "user" in roles
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# handle_chat (async generator)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_chat_yields_thinking_first():
    with patch("connoisseur.mcp.app.chat_with_agent", new=AsyncMock(return_value="Final answer.")):
        yielded = []
        async for history in handle_chat("What restaurants are near me?", []):
            yielded.append(history)

    # First yield should include the "Thinking..." placeholder
    assert len(yielded) >= 1
    thinking_history = yielded[0]
    contents = [m.get("content") for m in thinking_history]
    assert "Thinking..." in contents


@pytest.mark.asyncio
async def test_handle_chat_yields_final_answer():
    with patch("connoisseur.mcp.app.chat_with_agent", new=AsyncMock(return_value="Final answer.")):
        yielded = []
        async for history in handle_chat("What restaurants are near me?", []):
            yielded.append(history)

    final_history = yielded[-1]
    contents = [m.get("content") for m in final_history]
    assert "Final answer." in contents


@pytest.mark.asyncio
async def test_handle_chat_empty_message_yields_history():
    initial_history = [{"role": "user", "content": "hi"}]
    yielded = []
    async for h in handle_chat("", initial_history):
        yielded.append(h)
    # Empty message should yield the unchanged history and return
    assert len(yielded) == 1
    assert yielded[0] == initial_history


@pytest.mark.asyncio
async def test_handle_chat_whitespace_message_yields_history():
    yielded = []
    async for h in handle_chat("   ", []):
        yielded.append(h)
    assert len(yielded) == 1
