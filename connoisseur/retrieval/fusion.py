"""M2L3 — Cross-modal score normalisation, weighted fusion, and reranking."""
from __future__ import annotations

import numpy as np

from connoisseur.config import ARTICLE_COLLECTION, CHROMA_DIR, IMAGE_COLLECTION
from connoisseur.retrieval.retrieve import _get_article_db, _get_image_db, _unwrap
from connoisseur.retrieval.embeddings import embed_texts, embed_query_clip_text


# ── Score utilities ───────────────────────────────────────────────────────────

def _to_similarity(dists: list | np.ndarray) -> np.ndarray:
    """Convert 'smaller is better' distance to 'larger is better' similarity."""
    return 1.0 - np.array(dists, dtype=np.float32)


def _minmax(x: list | np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns ones array for constant input."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


# ── Per-modality retrieval (with similarity output) ───────────────────────────

def retrieve_articles(
    query: str,
    k: int = 5,
    where: dict | None = None,
) -> tuple[list, list, list, np.ndarray]:
    """Text → article retrieval; returns distances converted to similarities."""
    q_vec = embed_texts([query])[0]
    res = _get_article_db()._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    return ids, docs, metas, _to_similarity(dists)


def retrieve_images_by_text(
    query: str,
    k: int = 5,
    where: dict | None = None,
) -> tuple[list, list, list, np.ndarray]:
    """Text → image retrieval via CLIP; returns distances converted to similarities."""
    q_vec = embed_query_clip_text(query)
    res = _get_image_db()._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    return ids, docs, metas, _to_similarity(dists)


# ── Fusion ────────────────────────────────────────────────────────────────────

def fuse_rank(
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict | None = None,
    where_img: dict | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Retrieve from both modalities, normalise scores, and return a fused ranking.

    Each result row contains:
        modality, id, cuisine, location, source,
        text_score, img_score, fused, snippet
    """
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(query, k=k_text, where=where_text)
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(query, k=k_img, where=where_img)

    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    rows: list[dict] = []
    for j in range(len(t_ids)):
        meta = t_metas[j] if isinstance(t_metas[j], dict) else {}
        rows.append({
            "modality": "article",
            "id": meta.get("doc_id", t_ids[j]),
            "cuisine": meta.get("cuisine", "N/A"),
            "location": meta.get("location", "N/A"),
            "source": meta.get("source", "N/A"),
            "text_score": float(t_norm[j]),
            "img_score": 0.0,
            "fused": float(w_text * t_norm[j]),
            "snippet": (t_docs[j] or "").replace("\n", " ").strip(),
        })

    for j in range(len(i_ids)):
        meta = i_metas[j] if isinstance(i_metas[j], dict) else {}
        rows.append({
            "modality": "image",
            "id": meta.get("doc_id", i_ids[j]),
            "cuisine": meta.get("cuisine", "N/A"),
            "location": meta.get("location", "N/A"),
            "source": meta.get("source", "N/A"),
            "text_score": 0.0,
            "img_score": float(i_norm[j]),
            "fused": float(w_img * i_norm[j]),
            "snippet": (i_docs[j] or "").replace("\n", " ").strip(),
        })

    rows.sort(key=lambda r: r["fused"], reverse=True)
    top_n = max(0, min(int(top_n), len(rows)))
    return rows[:top_n]


def print_fused(rows: list[dict], title: str, max_chars: int = 90) -> None:
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{idx}] {r['modality']} | id={r['id']} | cuisine={r['cuisine']} | "
            f"location={r['location']} | fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )
        print(snippet)
