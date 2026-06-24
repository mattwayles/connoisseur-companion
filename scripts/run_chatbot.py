#!/usr/bin/env python3
"""M3L3 — Launch the Gradio multi-agent chatbot (port 7860)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connoisseur.agents.chatbot import launch

if __name__ == "__main__":
    launch(port=7860, share=False)
