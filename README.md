# Connoisseur Companion

An AI-powered multimodal restaurant and recipe recommendation system built on a multi-agent architecture with Claude, ChromaDB, and MCP.

## Overview

Connoisseur Companion processes a curated California restaurant dataset through a four-stage pipeline:

1. **Ingestion** — LLM-structured restaurant records + vision-captioned food images
2. **Retrieval** — Multimodal ChromaDB index (MiniLM text + CLIP image embeddings)
3. **Agents** — Six-agent ReAct workflow with parallel analysis phase
4. **MCP** — FastMCP server exposing restaurant data as tools and resources

Two Gradio UIs are provided: a multi-agent chatbot and an MCP host application.

## Architecture

```
connoisseur/
├── ingestion/
│   ├── models.py          # Pydantic Restaurant schema
│   ├── structure_text.py  # Raw paragraphs → structured JSON via LLM
│   ├── process_images.py  # Caption recipe/review images via vision LLM
│   └── data_manager.py    # Interactive CLI for CRUD on the restaurant DB
├── retrieval/
│   ├── embeddings.py      # MiniLM (text) + CLIP (image) embedding models
│   ├── build_index.py     # Build and persist ChromaDB collections
│   ├── retrieve.py        # Similarity search (text→article, image→image)
│   └── fusion.py          # Cross-modal score fusion and reranking
├── agents/
│   ├── configs.py         # Six agent persona definitions
│   ├── workflow.py        # Four-phase orchestrator (Phase 3 runs in parallel)
│   └── chatbot.py         # Gradio multi-agent chatbot UI
└── mcp/
    ├── server.py          # FastMCP server (1 resource, 3 tools)
    ├── client.py          # MCP client with sampling callback
    └── app.py             # Gradio MCP host with ReAct agent loop
```

### Agent Workflow

```
Phase 1 (sequential)   → User Profile Generator
Phase 2 (sequential)   → RAG Retriever (20 restaurant + 20 recipe candidates)
Phase 3 (parallel)     → Food Trend Analyst
                          Food Style Expert
                          Nutrition Expert
Phase 4 (sequential)   → Recommendation Expert (top-5 restaurants + top-5 recipes)
```

### MCP Tools

| Tool | Description |
|---|---|
| `get_restaurant_info` | Exact/partial name lookup with structured details |
| `recommend_by_vibe` | Two-pass vibe search (structured fields + raw text) |
| `get_review` | Full user review retrieval by restaurant name |

Resource: `culinary-map://california` — full California Culinary Map text.

## Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

## Installation

```bash
# Clone the repo
git clone https://github.com/mattwayles/connoisseur-companion.git
cd connoisseur-companion

# Install dependencies
pip install -e .
# or: pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your_key_here
```

## Data Setup

Raw data files are not included in the repository. Download them with the provided script (~205 MB total, including recipe images):

```bash
cd data
bash download.sh
```

This fetches:
- `California-Culinary-Map.txt` — 100+ California restaurant descriptions
- `Recipes.json` — Recipe dataset
- `Synthetic-User-Reviews.json` — Synthetic user reviews with image URLs
- `synthetic_recipe_images/` — PNG recipe images (~205 MB)

## Pipeline

Run the scripts in order to build the full system from raw data:

```bash
# 1. Structure restaurant text into JSON records
python scripts/run_structure_text.py

# 2. Caption recipe and review images via vision LLM
python scripts/run_process_images.py

# 3. Build the multimodal ChromaDB vector index
python scripts/run_build_index.py
```

The vector database is persisted to `~/chroma_connoisseur/`.

## Running the Apps

### Multi-Agent Chatbot (port 7860)

```bash
python scripts/run_chatbot.py
```

Opens a Gradio UI where you can ask for restaurant and recipe recommendations. Uses a six-agent pipeline to profile your preferences, retrieve candidates, run parallel analysis, and synthesise personalised recommendations.

### MCP Host Application (port 7861)

```bash
python scripts/run_mcp_app.py
```

Opens a Gradio UI backed by a FastMCP server and an Anthropic ReAct agent loop. The agent calls MCP tools to look up restaurants by name, vibe, or review.

### MCP Server (standalone)

```bash
python scripts/run_mcp_server.py
```

Starts the FastMCP server over stdio — useful for connecting external MCP clients.

### Other Scripts

```bash
python scripts/run_data_manager.py        # Interactive CLI to browse/add/edit/delete restaurants
python scripts/run_data_manager.py --test # Run unit tests
python scripts/run_retrieve.py            # Demo similarity retrieval
python scripts/run_fusion.py              # Demo cross-modal score fusion
python scripts/run_mcp_client.py          # Demo MCP client tools
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (Haiku / Sonnet) |
| Vector DB | ChromaDB |
| Text Embeddings | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Image Embeddings | `openai/clip-vit-base-patch32` |
| Retrieval | LangChain + Chroma |
| UI | Gradio |
| MCP | FastMCP + Anthropic MCP SDK |
| Validation | Pydantic v2 |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude inference |
