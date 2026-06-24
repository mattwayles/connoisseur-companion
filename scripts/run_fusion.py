#!/usr/bin/env python3
"""M2L3 — Demo multimodal similarity fusion and retrieval reranking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connoisseur.retrieval.fusion import fuse_rank, print_fused
from connoisseur.config import CHROMA_DIR

if __name__ == "__main__":
    if not CHROMA_DIR.exists():
        print("⚠️  Vector DB not found. Run run_build_index.py first.")
        sys.exit(1)

    # Demo 1 — no filters, balanced weights
    rows = fuse_rank("cozy noodles with warm atmosphere", k_text=5, k_img=5, top_n=5)
    print_fused(rows, title="Demo 1 — Multimodal fusion (no filters)")
    print("✅ Demo 1 complete")

    # Demo 2 — with metadata filters
    rows2 = fuse_rank(
        "handmade pasta and romantic dinner",
        where_text={"location": "Pasadena"},
        where_img={"source": "recipe_image"},
        top_n=5,
    )
    print_fused(rows2, title="Demo 2 — Multimodal fusion (metadata filters)")
    print("✅ Demo 2 complete")

    q = "fresh sushi and minimalist presentation"

    # Demo 3A — text-heavy
    rows3a = fuse_rank(q, w_text=0.8, w_img=0.2,
                       where_text={"location": "Pasadena"},
                       where_img={"source": "recipe_image"}, top_n=5)
    print_fused(rows3a, title="Demo 3A — Text-heavy fusion (w_text=0.8, w_img=0.2)")

    # Demo 3B — image-heavy
    rows3b = fuse_rank(q, w_text=0.3, w_img=0.7,
                       where_text={"location": "Pasadena"},
                       where_img={"source": "recipe_image"}, top_n=5)
    print_fused(rows3b, title="Demo 3B — Image-heavy fusion (w_text=0.3, w_img=0.7)")

    print("✅ Demo 3 complete")
    print("\n🎉 Multimodal Similarity Fusion and Retrieval Ranking COMPLETE")
