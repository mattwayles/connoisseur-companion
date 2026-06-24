"""M4L3 — Full MCP host application: Gradio UI + Anthropic ReAct agent loop."""
from __future__ import annotations

import asyncio
from pathlib import Path

import gradio as gr
from anthropic import Anthropic
from fastmcp.client import Client, PythonStdioTransport

SERVER_SCRIPT = str(Path(__file__).parent / "server.py")

SYSTEM_PROMPT = """You are Connoisseur Companion, an AI guide to California's restaurant scene.

You have access to a curated database of California restaurants and can use these tools:
- get_restaurant_info: Look up a specific restaurant by name to retrieve details such as
  cuisine type, rating, price range, vibe, and signature dishes.
- recommend_by_vibe: Find restaurants that match a mood or atmosphere keyword
  (e.g., 'moody', 'romantic', 'zen', 'sun-drenched', 'cozy').
- get_review: Retrieve detailed user reviews for a specific restaurant.

Be warm, enthusiastic, and knowledgeable. Always use your tools to provide accurate,
grounded information about California's culinary landscape."""

_anthropic = Anthropic()


# ── ReAct agent loop ──────────────────────────────────────────────────────────

async def chat_with_agent(user_message: str, history: list[dict]) -> str:
    """Connect to the MCP server, discover tools, and run a ReAct loop.

    The LLM decides which tools to call, executes them through the MCP server,
    and repeats until it produces a final text response.
    """
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        mcp_tools = await client.list_tools()

        # Convert MCP tool schemas to Anthropic tool format
        anthropic_tools = [
            {
                "name":         t.name,
                "description":  t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in mcp_tools
        ]

        # Reconstruct conversation history
        messages: list[dict] = []
        for msg in history:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content and content != "Thinking...":
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        # ReAct loop — iterate until the model stops calling tools
        for _ in range(10):
            response = _anthropic.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=anthropic_tools,
                messages=messages,
            )

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return "I couldn't generate a response. Please try again."

            if response.stop_reason == "tool_use":
                tool_results: list[dict] = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await client.call_tool(block.name, block.input)
                        tool_output = (
                            " ".join(
                                item.text if hasattr(item, "text") else str(item)
                                for item in result.content
                            )
                            if result.content
                            else "(no result)"
                        )
                        tool_results.append(
                            {
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     tool_output,
                            }
                        )
                messages.append({"role": "user", "content": tool_results})

    return "I wasn't able to complete that request. Please try again."


# ── Gradio event handler ──────────────────────────────────────────────────────

async def handle_chat(user_message: str, history: list[dict]):
    """Stream a 'Thinking...' placeholder then replace it with the agent's response."""
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history
        return

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield history

    response_text = await chat_with_agent(user_message, history[:-2])
    history[-1] = {"role": "assistant", "content": response_text}
    yield history


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Connoisseur Companion", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Connoisseur Companion\n"
            "Your AI guide to California's restaurant scene. "
            "Ask me about restaurants by name, cuisine, or vibe!"
        )

        chatbot   = gr.Chatbot(height=500, type="messages")
        msg_input = gr.Textbox(
            label="Ask about restaurants",
            placeholder='"Find me a moody spot in DTLA" or "Tell me about The Gilded Artichoke"',
        )

        with gr.Row():
            btn1 = gr.Button("Find moody restaurants",          size="sm")
            btn2 = gr.Button("Show me romantic dining options", size="sm")
            btn3 = gr.Button("Best sushi spots in California?", size="sm")

        msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot])
        msg_input.submit(lambda: "", None, msg_input)

        btn1.click(handle_chat, [gr.State("Find me some moody restaurants"),                          chatbot], [chatbot])
        btn2.click(handle_chat, [gr.State("Show me romantic dining options"),                         chatbot], [chatbot])
        btn3.click(handle_chat, [gr.State("What are the best sushi spots in California?"),            chatbot], [chatbot])

    return demo


def launch(port: int = 7861, share: bool = False) -> None:
    demo = build_app()
    print("Starting Connoisseur Companion MCP App...")
    demo.launch(server_port=port, share=share)


if __name__ == "__main__":
    launch(share=True)
