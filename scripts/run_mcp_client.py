#!/usr/bin/env python3
"""M4L2 — Run MCP client demos and connection verification."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connoisseur.mcp.client import main

if __name__ == "__main__":
    asyncio.run(main())
