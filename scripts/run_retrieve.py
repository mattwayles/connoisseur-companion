#!/usr/bin/env python3
"""M2L2 — Demo similarity retrieval with optional metadata filters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connoisseur.retrieval.retrieve import retrieve_articles, retrieve_images_by_image, print_hits
from connoisseur.config import CHROMA_DIR, IMAGE_DIR

if __name__ == "__main__":
    if not CHROMA_DIR.exists():
        print("⚠️  Vector DB not found. Run run_build_index.py first.")
        sys.exit(1)

    # Demo 1 — plain text search
    q = "cozy restaurant with noodles and warm atmosphere"
    ids, docs, metas, dists = retrieve_articles(q, k=5)
    print_hits(ids, docs, metas, dists, title="Demo 1 — Article similarity search (no filter)")
    print("✅ Demo 1 complete")

    # Demo 2 — with metadata filter
    q2 = "handmade pasta and romantic dinner"
    ids2, docs2, metas2, dists2 = retrieve_articles(q2, k=5, where={"location": "Pasadena"})
    print_hits(ids2, docs2, metas2, dists2, title="Demo 2 — Article similarity search + location filter")
    print("✅ Demo 2 complete")

    # Demo 3 — image-to-image search (if images exist)
    import glob
    image_paths = sorted(glob.glob(str(IMAGE_DIR / "*.png")))
    if image_paths:
        query_img = image_paths[0]
        ids3, docs3, metas3, dists3 = retrieve_images_by_image(query_img, k=5)
        print_hits(ids3, docs3, metas3, dists3, title=f"Demo 3 — Image search (query: {Path(query_img).name})")
        print("✅ Demo 3 complete")

    print("\n🎉 Similarity Retrieval with Metadata Filtering COMPLETE")
