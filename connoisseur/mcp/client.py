"""M4L2 — MCP client with roots declaration, sampling callback, and tool demos."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    Root,
    TextContent,
)

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
PROJECT_DIR   = Path(__file__).parent.resolve()

server_params = StdioServerParameters(
    command="python",
    args=[SERVER_SCRIPT],
)

_anthropic = Anthropic()


# ── Roots callback ────────────────────────────────────────────────────────────

def list_roots() -> list[Root]:
    """Declare the project directory as the only permitted filesystem root."""
    return [Root(uri=f"file://{PROJECT_DIR}", name=PROJECT_DIR.name)]


# ── Sampling callback ─────────────────────────────────────────────────────────

async def handle_sampling(params: CreateMessageRequestParams) -> CreateMessageResult:
    """Proxy an LLM call requested by the server through the Anthropic API."""
    prompt = params.messages[0].content.text
    print(f"\n[Sampling] prompt preview: {prompt[:120]}...")

    response = _anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=params.maxTokens or 200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    print(f"[Sampling] response: {text[:80]}...")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=text),
        model="claude-haiku-4-5-20251001",
    )


# ── Session helper ────────────────────────────────────────────────────────────

async def call_tool(tool_name: str, arguments: dict) -> dict:
    """Open a session, call a single tool, and return the parsed JSON result."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return json.loads(result.content[0].text)


# ── Connection verification ───────────────────────────────────────────────────

async def verify_connection() -> None:
    print("=" * 60)
    print("MCP Connection Verification")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tool_names   = [t.name for t in tools_result.tools]

            print("--- START SCREENSHOT ---")
            print(f"\nDiscovered {len(tool_names)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {(tool.description or '')[:80]}...")

            assert "get_restaurant_info" in tool_names, "FAIL: get_restaurant_info missing"
            assert "recommend_by_vibe"   in tool_names, "FAIL: recommend_by_vibe missing"
            assert "get_review"          in tool_names, "FAIL: get_review missing"
            print("\nAll required tools verified!")

            resources_result = await session.list_resources()
            print(f"\nDiscovered {len(resources_result.resources)} resources:")
            for r in resources_result.resources:
                print(f"  - {r.uri}: {r.name}")

            roots = list_roots()
            print(f"\nConfigured {len(roots)} roots:")
            for root in roots:
                print(f"  - {root.name}: {root.uri}")
            print("--- END SCREENSHOT ---")


# ── Demo functions ────────────────────────────────────────────────────────────

async def demo_get_restaurant_info() -> None:
    print("\n" + "-" * 60)
    print("Demo: get_restaurant_info('Gilded')")
    print("-" * 60)
    data = await call_tool("get_restaurant_info", {"restaurant_name": "Gilded"})
    print(json.dumps(data, indent=2))


async def demo_recommend_by_vibe() -> None:
    print("\n" + "-" * 60)
    print("Demo: recommend_by_vibe('moody')")
    print("-" * 60)
    data = await call_tool("recommend_by_vibe", {"vibe": "moody"})
    print(f"Vibe searched: {data['vibe_searched']}")
    print(f"Structured matches: {len(data['structured_matches'])}")
    for m in data["structured_matches"][:3]:
        print(f"  - {m['name']} ({m.get('food_style', 'N/A')}) — {m.get('rating')}/5")
    print(f"Raw text excerpts: {len(data['raw_text_excerpts'])}")


async def demo_get_review() -> None:
    print("\n" + "-" * 60)
    print("Demo: get_review('Artichoke')")
    print("-" * 60)
    data = await call_tool("get_review", {"restaurant_name": "Artichoke"})
    print(json.dumps(data, indent=2))


# ── Main entry point ──────────────────────────────────────────────────────────

async def main() -> None:
    await demo_get_restaurant_info()
    await demo_recommend_by_vibe()
    await demo_get_review()
    await verify_connection()


if __name__ == "__main__":
    asyncio.run(main())
