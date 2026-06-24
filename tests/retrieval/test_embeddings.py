"""Tests for connoisseur.retrieval.embeddings — singleton model wrappers."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import connoisseur.retrieval.embeddings as emb_module


# ---------------------------------------------------------------------------
# Fixture: reset singletons so each test starts with a clean slate
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    emb_module._text_model = None
    emb_module._clip_model = None
    emb_module._clip_processor = None
    yield
    emb_module._text_model = None
    emb_module._clip_model = None
    emb_module._clip_processor = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_sentence_transformer(n_texts=1, dim=384):
    model = MagicMock()
    model.encode.return_value = np.ones((n_texts, dim), dtype=np.float32)
    return model


def _mock_clip_model(n=1, dim=512):
    import torch
    model = MagicMock()
    feats = torch.ones(n, dim)
    model.get_image_features.return_value = feats
    model.get_text_features.return_value = feats
    return model


def _mock_clip_processor():
    import torch
    proc = MagicMock()
    proc.return_value = {
        "pixel_values": torch.zeros(1, 3, 224, 224),
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return proc


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

@patch("connoisseur.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_returns_ndarray(MockST):
    MockST.return_value = _mock_sentence_transformer(1)
    result = emb_module.embed_texts(["hello world"])
    assert isinstance(result, np.ndarray)


@patch("connoisseur.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_shape(MockST):
    MockST.return_value = _mock_sentence_transformer(3)
    result = emb_module.embed_texts(["a", "b", "c"])
    assert result.shape == (3, 384)


@patch("connoisseur.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_dtype_is_float32(MockST):
    MockST.return_value = _mock_sentence_transformer(2)
    result = emb_module.embed_texts(["x", "y"])
    assert result.dtype == np.float32


@patch("connoisseur.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_loads_model_once(MockST):
    MockST.return_value = _mock_sentence_transformer(1)
    emb_module.embed_texts(["a"])
    emb_module.embed_texts(["b"])
    MockST.assert_called_once()


@patch("connoisseur.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_passes_normalize_flag(MockST):
    model = _mock_sentence_transformer(1)
    MockST.return_value = model
    emb_module.embed_texts(["test"])
    _, kwargs = model.encode.call_args
    assert kwargs.get("normalize_embeddings") is True


# ---------------------------------------------------------------------------
# embed_images
# ---------------------------------------------------------------------------

@patch("connoisseur.retrieval.embeddings.CLIPProcessor.from_pretrained")
@patch("connoisseur.retrieval.embeddings.CLIPModel.from_pretrained")
@patch("connoisseur.retrieval.embeddings.Image.open")
def test_embed_images_returns_ndarray(mock_open, MockCLIPModel, MockCLIPProc):
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_open.return_value = mock_img

    clip_model = _mock_clip_model(1)
    MockCLIPModel.return_value = clip_model

    proc = _mock_clip_processor()
    MockCLIPProc.return_value = proc

    result = emb_module.embed_images(["fake/path.png"])
    assert isinstance(result, np.ndarray)


@patch("connoisseur.retrieval.embeddings.CLIPProcessor.from_pretrained")
@patch("connoisseur.retrieval.embeddings.CLIPModel.from_pretrained")
@patch("connoisseur.retrieval.embeddings.Image.open")
def test_embed_images_loads_clip_once(mock_open, MockCLIPModel, MockCLIPProc):
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_open.return_value = mock_img

    MockCLIPModel.return_value = _mock_clip_model(1)
    MockCLIPProc.return_value = _mock_clip_processor()

    emb_module.embed_images(["a.png"])
    emb_module.embed_images(["b.png"])

    MockCLIPModel.assert_called_once()
    MockCLIPProc.assert_called_once()


# ---------------------------------------------------------------------------
# embed_query_clip_text
# ---------------------------------------------------------------------------

@patch("connoisseur.retrieval.embeddings.CLIPProcessor.from_pretrained")
@patch("connoisseur.retrieval.embeddings.CLIPModel.from_pretrained")
def test_embed_query_clip_text_returns_1d_array(MockCLIPModel, MockCLIPProc):
    MockCLIPModel.return_value = _mock_clip_model(1, 512)
    MockCLIPProc.return_value = _mock_clip_processor()

    result = emb_module.embed_query_clip_text("spicy ramen")
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


@patch("connoisseur.retrieval.embeddings.CLIPProcessor.from_pretrained")
@patch("connoisseur.retrieval.embeddings.CLIPModel.from_pretrained")
def test_embed_query_clip_text_dtype_is_float32(MockCLIPModel, MockCLIPProc):
    MockCLIPModel.return_value = _mock_clip_model(1, 512)
    MockCLIPProc.return_value = _mock_clip_processor()

    result = emb_module.embed_query_clip_text("query")
    assert result.dtype == np.float32
