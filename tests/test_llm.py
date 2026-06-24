"""Tests for connoisseur.llm — Anthropic client singleton and call wrappers."""
import pytest
from unittest.mock import MagicMock, patch, call
import connoisseur.llm as llm_module
from connoisseur.config import CLAUDE_HAIKU, CLAUDE_SONNET


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure the global client is None before and after every test."""
    original = llm_module._client
    llm_module._client = None
    yield
    llm_module._client = original


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------

@patch("connoisseur.llm.anthropic.Anthropic")
def test_get_client_creates_instance(MockAnthropic):
    instance = MagicMock()
    MockAnthropic.return_value = instance

    result = llm_module.get_client()

    MockAnthropic.assert_called_once()
    assert result is instance


@patch("connoisseur.llm.anthropic.Anthropic")
def test_get_client_is_singleton(MockAnthropic):
    instance = MagicMock()
    MockAnthropic.return_value = instance

    c1 = llm_module.get_client()
    c2 = llm_module.get_client()

    assert c1 is c2
    MockAnthropic.assert_called_once()  # constructed only once


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------

@patch("connoisseur.llm.anthropic.Anthropic")
def test_call_llm_returns_text(MockAnthropic):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("hello world")
    MockAnthropic.return_value = client

    result = llm_module.call_llm("sys", "user prompt")

    assert result == "hello world"


@patch("connoisseur.llm.anthropic.Anthropic")
def test_call_llm_passes_model_and_tokens(MockAnthropic):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("ok")
    MockAnthropic.return_value = client

    llm_module.call_llm("sys", "prompt", model=CLAUDE_SONNET, max_tokens=128)

    kwargs = client.messages.create.call_args[1]
    assert kwargs["model"] == CLAUDE_SONNET
    assert kwargs["max_tokens"] == 128
    assert kwargs["system"] == "sys"


@patch("connoisseur.llm.anthropic.Anthropic")
def test_call_llm_default_model_is_haiku(MockAnthropic):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("ok")
    MockAnthropic.return_value = client

    llm_module.call_llm("sys", "prompt")

    kwargs = client.messages.create.call_args[1]
    assert kwargs["model"] == CLAUDE_HAIKU


@patch("connoisseur.llm.anthropic.Anthropic")
def test_call_llm_message_structure(MockAnthropic):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("ok")
    MockAnthropic.return_value = client

    llm_module.call_llm("my system", "my prompt")

    messages = client.messages.create.call_args[1]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "my prompt"


# ---------------------------------------------------------------------------
# call_vision_llm
# ---------------------------------------------------------------------------

@patch("connoisseur.llm.anthropic.Anthropic")
def test_call_vision_llm_returns_text(MockAnthropic):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("a salmon dish")
    MockAnthropic.return_value = client

    result = llm_module.call_vision_llm("sys", "describe", "b64data", "image/jpeg")

    assert result == "a salmon dish"


@patch("connoisseur.llm.anthropic.Anthropic")
def test_call_vision_llm_includes_image_block(MockAnthropic):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("ok")
    MockAnthropic.return_value = client

    llm_module.call_vision_llm("sys", "describe", "b64data", "image/png")

    messages = client.messages.create.call_args[1]["messages"]
    content = messages[0]["content"]
    image_block = content[0]

    assert image_block["type"] == "image"
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["data"] == "b64data"
    assert image_block["source"]["media_type"] == "image/png"


@patch("connoisseur.llm.anthropic.Anthropic")
def test_call_vision_llm_includes_text_block(MockAnthropic):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("ok")
    MockAnthropic.return_value = client

    llm_module.call_vision_llm("sys", "my question", "b64")

    messages = client.messages.create.call_args[1]["messages"]
    text_block = messages[0]["content"][1]
    assert text_block["type"] == "text"
    assert text_block["text"] == "my question"


@patch("connoisseur.llm.anthropic.Anthropic")
def test_call_vision_llm_default_media_type(MockAnthropic):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("ok")
    MockAnthropic.return_value = client

    llm_module.call_vision_llm("sys", "prompt", "b64data")

    messages = client.messages.create.call_args[1]["messages"]
    image_block = messages[0]["content"][0]
    assert image_block["source"]["media_type"] == "image/png"
