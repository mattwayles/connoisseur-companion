"""M2 — Singleton embedding models for text (MiniLM) and images (CLIP)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cpu"
_text_model: SentenceTransformer | None = None
_clip_model: CLIPModel | None = None
_clip_processor: CLIPProcessor | None = None


def _get_text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_model


def _get_clip() -> tuple[CLIPModel, CLIPProcessor]:
    global _clip_model, _clip_processor
    if _clip_model is None:
        name = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(name).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(name, use_fast=True)
        _clip_model.eval()
    return _clip_model, _clip_processor


# ── Public embedding functions ────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """384-d L2-normalised embeddings for a list of strings."""
    return _get_text_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)


@torch.no_grad()
def embed_images(paths: list, batch_size: int = 16) -> np.ndarray:
    """512-d L2-normalised CLIP image embeddings for a list of file paths."""
    model, processor = _get_clip()
    vecs: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)


@torch.no_grad()
def embed_query_clip_text(query: str) -> np.ndarray:
    """512-d L2-normalised CLIP text embedding for a single query string."""
    model, processor = _get_clip()
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype(np.float32)
