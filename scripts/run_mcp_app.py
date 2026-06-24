#!/usr/bin/env python3
"""M4L3 — Launch the full MCP Gradio host application (port 7861)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connoisseur.mcp.app import launch

if __name__ == "__main__":
    launch(port=7861, share=False)
