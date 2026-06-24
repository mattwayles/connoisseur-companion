"""M2L1 — Build and persist multimodal ChromaDB vector indexes."""
import glob
import json
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from connoisseur.config import (
    ARTICLE_COLLECTION,
    CHROMA_DIR,
    IMAGE_COLLECTION,
    IMAGE_DIR,
    RECIPE_DATA_PATH,
    RESTAURANT_DATA_PATH,
)
from connoisseur.retrieval.embeddings import embed_images, embed_texts


# ── Document builders ─────────────────────────────────────────────────────────

def _build_article_docs(restaurants: list[dict]) -> list[Document]:
    docs: list[Document] = []
    for i, r in enumerate(restaurants):
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {r.get('food_style', '')}\n"
            f"Location: {r.get('location', '')}"
        )
        docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "doc_id": f"rest_{i}",
                    "cuisine": r.get("food_style"),
                    "location": r.get("location"),
                    "source": "restaurant",
                },
            )
        )
    return docs


def _build_image_docs(image_paths: list[str], recipes: list[dict]) -> list[Document]:
    docs: list[Document] = []
    for i, (p, rec) in enumerate(zip(image_paths, recipes)):
        docs.append(
            Document(
                page_content=rec.get("name", f"recipe image {i}"),
                metadata={
                    "doc_id": f"img_{i}",
                    "image_path": str(p),
                    "source": "recipe_image",
                    "recipe_id": rec.get("id"),
                    "cuisine": rec.get("cuisine"),
                },
            )
        )
    return docs


# ── Main builder ──────────────────────────────────────────────────────────────

def build_index(reset: bool = True) -> tuple[Chroma, Chroma]:
    """Build article and image Chroma collections and return them.

    Args:
        reset: If True, wipe existing DB before rebuilding.

    Returns:
        (article_db, image_db) Chroma instances.
    """
    with open(RESTAURANT_DATA_PATH, encoding="utf-8") as f:
        restaurants: list[dict] = json.load(f)
    with open(RECIPE_DATA_PATH, encoding="utf-8") as f:
        recipes: list[dict] = json.load(f)

    # Discover recipe images (flat dir or nested)
    image_paths = sorted(glob.glob(str(IMAGE_DIR / "*.png")))
    if not image_paths:
        image_paths = sorted(glob.glob(str(IMAGE_DIR / "**" / "*.png"), recursive=True))

    article_docs = _build_article_docs(restaurants)
    image_docs = _build_image_docs(image_paths, recipes)

    db_dir = str(CHROMA_DIR)
    if reset and os.path.isdir(db_dir):
        shutil.rmtree(db_dir)

    # ── Article collection ────────────────────────────────────────────────────
    A = embed_texts([d.page_content for d in article_docs])
    article_db = Chroma(collection_name=ARTICLE_COLLECTION, persist_directory=db_dir)
    article_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in article_docs],
        embeddings=A.tolist(),
        documents=[d.page_content for d in article_docs],
        metadatas=[d.metadata for d in article_docs],
    )
    print(f"✅ Article DB: {len(article_docs)} documents")

    # ── Image collection ──────────────────────────────────────────────────────
    image_db = Chroma(collection_name=IMAGE_COLLECTION, persist_directory=db_dir)
    if image_docs:
        V = embed_images([d.metadata["image_path"] for d in image_docs])
        image_db._collection.upsert(
            ids=[d.metadata["doc_id"] for d in image_docs],
            embeddings=V.tolist(),
            documents=[d.page_content for d in image_docs],
            metadatas=[d.metadata for d in image_docs],
        )
        print(f"✅ Image DB: {len(image_docs)} documents")
    else:
        print("⚠️  No images found — image DB will be empty")

    print("🎉 Multimodal Vector Index Construction COMPLETE")
    return article_db, image_db
