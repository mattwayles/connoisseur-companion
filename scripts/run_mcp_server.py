#!/usr/bin/env python3
"""M4L1 — Start the FastMCP server standalone (stdio transport)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import triggers mcp.run() via __main__ guard in server.py
import subprocess

server = Path(__file__).parent.parent / "connoisseur" / "mcp" / "server.py"
subprocess.run([sys.executable, str(server)], check=True)
