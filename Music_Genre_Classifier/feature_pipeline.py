"""
feature_pipeline.py
───────────────────
Orchestrates feature extraction for the full dataset and caches results
to disk so subsequent runs skip the expensive audio processing step.

Output cached object (pickle):
    {
      "audio_features"  : np.ndarray  (N, D_audio),
      "mel_images"      : np.ndarray  (N, 1, N_MELS, MEL_IMG_FRAMES),
      "lyric_embeddings": np.ndarray  (N, D_lyrics),
      "lyric_texts"     : List[str],
      "labels"          : np.ndarray  (N,)  int encoded,
      "song_ids"        : List[str],
      "genres"          : List[str],
      "label_encoder"   : sklearn LabelEncoder,
    }
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import config
from data_loader  import build_manifest
from audio_features import extract_audio_features, extract_mel_spectrogram_image
from lyric_features import load_lyrics, embed_lyrics_batch


def build_or_load_features(force_rebuild: bool = False) -> dict:
    """
    If cache exists (and force_rebuild=False), load from disk.
    Otherwise extract everything and save.
    """
    if not force_rebuild and os.path.isfile(config.FEATURE_CACHE):
        print(f"[feature_pipeline] Loading cached features from {config.FEATURE_CACHE}")
        with open(config.FEATURE_CACHE, "rb") as f:
            cached = pickle.load(f)
        required = {"audio_features", "mel_images", "lyric_embeddings", "lyric_texts", "labels", "label_encoder"}
        if required.issubset(cached.keys()):
            return cached
        print("[feature_pipeline] Cache missing new fields (mel_images). Rebuilding …")

    print("[feature_pipeline] Building features from scratch …")

    # ── 1. manifest ───────────────────────────────────────────────────────────
    df = build_manifest(
        dataset_root=config.MAIN_DATASET_ROOT if os.path.isdir(config.MAIN_DATASET_ROOT) else None,
        max_per_genre=config.TRAIN_SONGS_PER_GENRE,
    )

    # ── 2. label encoding ─────────────────────────────────────────────────────
    le = LabelEncoder()
    labels = le.fit_transform(df["genre"].values)

    # ── 3. audio features ─────────────────────────────────────────────────────
    print("[feature_pipeline] Extracting audio features …")
    audio_feats = []
    mel_images = []
    for path in tqdm(df["audio_path"], desc="Audio", unit="song"):
        audio_feats.append(extract_audio_features(path))
        mel_images.append(extract_mel_spectrogram_image(path))
    audio_feats = np.vstack(audio_feats).astype(np.float32)
    mel_images = np.stack(mel_images).astype(np.float32)
    print(f"  Audio feature matrix: {audio_feats.shape}")
    print(f"  Mel image tensor    : {mel_images.shape}")

    # ── 4. lyric texts ────────────────────────────────────────────────────────
    print("[feature_pipeline] Loading lyrics …")
    lyric_texts = [load_lyrics(p) for p in tqdm(df["lyrics_path"], desc="Lyrics", unit="song")]

    # ── 5. lyric embeddings ───────────────────────────────────────────────────
    print("[feature_pipeline] Computing lyric embeddings …")
    lyric_embeds = embed_lyrics_batch(lyric_texts, show_progress=True)
    print(f"  Lyric embedding matrix: {lyric_embeds.shape}")

    # ── 6. assemble & cache ───────────────────────────────────────────────────
    data = {
        "audio_features"   : audio_feats,
        "mel_images"       : mel_images,
        "lyric_embeddings" : lyric_embeds,
        "lyric_texts"      : lyric_texts,
        "labels"           : labels,
        "song_ids"         : df["song_id"].tolist(),
        "genres"           : df["genre"].tolist(),
        "label_encoder"    : le,
    }

    os.makedirs(os.path.dirname(config.FEATURE_CACHE), exist_ok=True)
    with open(config.FEATURE_CACHE, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[feature_pipeline] Features cached to {config.FEATURE_CACHE}")

    return data


if __name__ == "__main__":
    data = build_or_load_features(force_rebuild=False)
    print("\nSummary:")
    print(f"  Audio features  : {data['audio_features'].shape}")
    print(f"  Lyric embeddings: {data['lyric_embeddings'].shape}")
    print(f"  Labels          : {data['labels'].shape}  classes={set(data['labels'])}")
