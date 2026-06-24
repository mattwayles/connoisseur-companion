"""M2L2 — Similarity retrieval with optional metadata filtering."""
from __future__ import annotations

import numpy as np
from langchain_chroma import Chroma

from connoisseur.config import ARTICLE_COLLECTION, CHROMA_DIR, IMAGE_COLLECTION
from connoisseur.retrieval.embeddings import embed_images, embed_texts

_article_db: Chroma | None = None
_image_db: Chroma | None = None


def _get_article_db() -> Chroma:
    global _article_db
    if _article_db is None:
        _article_db = Chroma(
            collection_name=ARTICLE_COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )
    return _article_db


def _get_image_db() -> Chroma:
    global _image_db
    if _image_db is None:
        _image_db = Chroma(
            collection_name=IMAGE_COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )
    return _image_db


def _unwrap(res: dict) -> tuple[list, list, list, list]:
    """Unwrap Chroma's nested list-of-lists for a single-query result."""
    return (
        res.get("ids", [[]])[0],
        res.get("documents", [[]])[0],
        res.get("metadatas", [[]])[0],
        res.get("distances", [[]])[0],
    )


# ── Retrieval functions ───────────────────────────────────────────────────────

def retrieve_articles(
    query: str,
    k: int = 5,
    where: dict | None = None,
) -> tuple[list, list, list, list]:
    """Text → article similarity search.

    Returns (ids, docs, metas, distances).
    """
    q_vec = embed_texts([query])[0]
    res = _get_article_db()._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)


def retrieve_images_by_image(
    query_image_path: str,
    k: int = 5,
    where: dict | None = None,
) -> tuple[list, list, list, list]:
    """Image → image similarity search.

    Returns (ids, docs, metas, distances).
    """
    q_vec = embed_images([query_image_path])[0]
    res = _get_image_db()._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)


def print_hits(
    ids: list,
    docs: list,
    metas: list,
    dists: list,
    title: str,
    max_chars: int = 180,
) -> None:
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        dist = float(dists[i]) if i < len(dists) else None
        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{i+1}] id={meta.get('doc_id','N/A')} | cuisine={meta.get('cuisine','N/A')} | "
            f"location={meta.get('location','N/A')} | source={meta.get('source','N/A')} | "
            f"distance={dist:.4f}"
        )
        print(snippet)
