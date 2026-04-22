"""
lyric_features.py
─────────────────
Extract text features from song lyrics (.txt files).

Two representations are produced:
  1. TF-IDF bag-of-words     → sparse (used for unimodal text baseline)
  2. Sentence-transformer embedding (all-MiniLM-L6-v2, 384 dims)
     → dense vector, used in multimodal fusion

The TF-IDF vectoriser is *fit on training data only* and applied to
val/test. The sentence-transformer is pre-trained and applied directly.
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

import config


# ─── Raw text loading ─────────────────────────────────────────────────────────

def load_lyrics(path: str) -> str:
    """Read a lyrics .txt file and return cleaned plain text."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        print(f"[lyric_features] ERROR reading {path}: {e}")
        return ""
    return _clean(text)


def _clean(text: str) -> str:
    """Lower-case, remove bracketed stage directions, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)      # remove [Chorus], [Verse 1] etc.
    text = re.sub(r"\(.*?\)", " ", text)      # remove (repeat) etc.
    text = re.sub(r"[^a-z\s']", " ", text)   # keep only letters, space, apostrophe
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── TF-IDF (for unimodal text baseline) ─────────────────────────────────────

def build_tfidf_pipeline() -> Pipeline:
    """Return an unfitted TF-IDF pipeline (fit on training corpus)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("norm", Normalizer()),
    ])


# ─── Sentence-Transformer embeddings (for multimodal fusion) ──────────────────

_sentence_model = None   # lazy load

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[lyric_features] Loading sentence-transformer: {config.LYRIC_EMBED_MODEL}")
        _sentence_model = SentenceTransformer(config.LYRIC_EMBED_MODEL)
    return _sentence_model


def embed_lyrics_batch(texts: List[str],
                        batch_size: int = 64,
                        show_progress: bool = True) -> np.ndarray:
    """
    Encode a list of lyric strings into dense vectors.
    Returns ndarray of shape (N, LYRIC_EMBED_DIM).
    """
    model = get_sentence_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise for cosine similarity
    )
    return embeddings.astype(np.float32)


if __name__ == "__main__":
    sample = "Look, if you had one shot or one opportunity to seize everything"
    cleaned = _clean(sample)
    print("Cleaned:", cleaned)
    emb = embed_lyrics_batch([cleaned], show_progress=False)
    print(f"Embedding shape: {emb.shape}")
