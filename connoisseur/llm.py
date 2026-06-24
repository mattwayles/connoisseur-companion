"""Singleton Anthropic client + convenience wrappers for text and vision calls."""
import anthropic
from connoisseur.config import CLAUDE_HAIKU

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def call_llm(
    system: str,
    prompt: str,
    model: str = CLAUDE_HAIKU,
    max_tokens: int = 4096,
) -> str:
    """Send a text prompt and return the assistant reply."""
    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_vision_llm(
    system: str,
    prompt: str,
    image_b64: str,
    media_type: str = "image/png",
    model: str = CLAUDE_HAIKU,
    max_tokens: int = 1024,
) -> str:
    """Send an image + text prompt and return the assistant reply."""
    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.content[0].text
