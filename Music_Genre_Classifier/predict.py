"""
predict.py
──────────
Inference script: predict the genre for a new (audio, lyrics) pair.

Usage examples:
    # Give both modalities (best accuracy):
    python predict.py --audio song.mp3 --lyrics song.txt

    # Audio only:
    python predict.py --audio song.mp3

    # Lyrics only (uses lyric embedding MLP):
    python predict.py --lyrics song.txt

    # Raw lyrics string directly:
    python predict.py --text "I used to roll the dice, fear the devil's eyes"

Output:
    Predicted genre : ROCK
    Confidence      : 87.3 %

    Top-3 predictions:
      1. ROCK            87.3 %
      2. METAL           8.1 %
      3. BLUES           2.4 %
"""

import argparse
import os
import pickle
import sys
import numpy as np

import config
from audio_features import extract_audio_features
from lyric_features  import load_lyrics, embed_lyrics_batch


# ─── Load models ──────────────────────────────────────────────────────────────

def _load(name: str):
    path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model '{path}' not found. Run train.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_label_encoder():
    cache_path = config.FEATURE_CACHE
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"Feature cache '{cache_path}' not found. Run train.py first.")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data["label_encoder"]


# ─── Core prediction ──────────────────────────────────────────────────────────

def predict(audio_path: str = None,
            lyrics_path: str = None,
            text: str = None,
            top_k: int = 3):
    """
    Predict genre from audio path, lyrics path, or raw text.
    Returns (predicted_label, confidence, top_k_list).
    """
    le = _load_label_encoder()

    if audio_path and (lyrics_path or text):
        # ── Multimodal (best) ─────────────────────────────────────────────────
        model = _load("late_fusion_learnable")

        audio_feats = extract_audio_features(audio_path).reshape(1, -1)

        raw_text = text if text else load_lyrics(lyrics_path)
        lyric_emb = embed_lyrics_batch([raw_text], show_progress=False)  # (1, 384)

        probs = model.predict_proba(audio_feats, lyric_emb)[0]

    elif audio_path:
        # ── Unimodal audio ────────────────────────────────────────────────────
        model = _load("audio_mlp")
        audio_feats = extract_audio_features(audio_path).reshape(1, -1)
        probs = model.predict_proba(audio_feats)[0]

    elif lyrics_path or text:
        # ── Unimodal lyrics ───────────────────────────────────────────────────
        model = _load("lyric_mlp")
        raw_text = text if text else load_lyrics(lyrics_path)
        lyric_emb = embed_lyrics_batch([raw_text], show_progress=False)
        probs = model.predict_proba(lyric_emb)[0]

    else:
        raise ValueError("Provide at least one of: --audio, --lyrics, --text")

    top_idx  = np.argsort(probs)[::-1]
    top_preds = [(le.classes_[i], float(probs[i])) for i in top_idx[:top_k]]

    predicted_label = top_preds[0][0]
    confidence      = top_preds[0][1]

    return predicted_label, confidence, top_preds


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict genre from audio and/or lyrics")
    parser.add_argument("--audio",   type=str, default=None, help="Path to .mp3 file")
    parser.add_argument("--lyrics",  type=str, default=None, help="Path to .txt lyrics file")
    parser.add_argument("--text",    type=str, default=None, help="Raw lyrics string")
    parser.add_argument("--top",     type=int, default=3,    help="Show top-N predictions")
    args = parser.parse_args()

    if not any([args.audio, args.lyrics, args.text]):
        parser.print_help()
        sys.exit(1)

    label, conf, top_k = predict(
        audio_path=args.audio,
        lyrics_path=args.lyrics,
        text=args.text,
        top_k=args.top,
    )

    print(f"\n  Predicted genre : {label}")
    print(f"  Confidence      : {conf*100:.1f} %")
    print(f"\n  Top-{args.top} predictions:")
    for rank, (genre, prob) in enumerate(top_k, 1):
        bar = "█" * int(prob * 30)
        print(f"    {rank}. {genre:<20} {prob*100:5.1f} %  {bar}")
    print()


if __name__ == "__main__":
    main()
