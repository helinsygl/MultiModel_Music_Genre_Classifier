"""
data_loader.py
──────────────
Extracts the dataset zip (if needed) and builds a DataFrame with
one row per song:  song_id | genre | audio_path | lyrics_path
"""

import os
import zipfile
import re
import pandas as pd
from pathlib import Path

import config


def extract_dataset(zip_path: str = config.DATASET_ZIP,
                    dest: str = ".") -> None:
    """Extract the zip archive once; skip if already extracted."""
    if os.path.isdir(config.DATASET_ROOT):
        print(f"[data_loader] Dataset already extracted at '{config.DATASET_ROOT}'")
        return
    print(f"[data_loader] Extracting '{zip_path}' …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print("[data_loader] Extraction complete.")


def _stem(path: str) -> str:
    """Return filename without extension, normalised to lower-case."""
    return Path(path).stem.lower()


def build_manifest(dataset_root: str = None, max_per_genre: int = None) -> pd.DataFrame:
    """
    Walk Audio/ and Lyrics/ directories and pair files by filename stem.
    Returns a DataFrame with columns:
        song_id, genre, audio_path, lyrics_path
    """
    if dataset_root is None:
        # Prefer structured local dataset folder if available
        if os.path.isdir(config.MAIN_DATASET_ROOT):
            dataset_root = config.MAIN_DATASET_ROOT
        else:
            extract_dataset()
            dataset_root = config.DATASET_ROOT

    audio_dir = os.path.join(dataset_root, "Audio")
    lyrics_dir = os.path.join(dataset_root, "Lyrics")

    records = []
    for genre in config.GENRES:
        audio_genre_dir  = os.path.join(audio_dir,  genre)
        lyrics_genre_dir = os.path.join(lyrics_dir, genre)

        if not os.path.isdir(audio_genre_dir):
            print(f"[data_loader] WARNING: missing audio dir  → {audio_genre_dir}")
            continue
        if not os.path.isdir(lyrics_genre_dir):
            print(f"[data_loader] WARNING: missing lyrics dir → {lyrics_genre_dir}")
            continue

        # Build stem → path maps
        audio_map  = {_stem(f): os.path.join(audio_genre_dir,  f)
                      for f in os.listdir(audio_genre_dir)  if f.endswith(".mp3")}
        lyrics_map = {_stem(f): os.path.join(lyrics_genre_dir, f)
                      for f in os.listdir(lyrics_genre_dir) if f.endswith(".txt")}

        common = set(audio_map) & set(lyrics_map)
        missing_audio  = set(lyrics_map) - set(audio_map)
        missing_lyrics = set(audio_map)  - set(lyrics_map)

        if missing_audio:
            print(f"[data_loader] {genre}: {len(missing_audio)} lyrics without audio — skipped")
        if missing_lyrics:
            print(f"[data_loader] {genre}: {len(missing_lyrics)} audio without lyrics — skipped")

        stems = sorted(common)
        if max_per_genre is not None:
            stems = stems[:max_per_genre]

        for stem in stems:
            records.append({
                "song_id"     : stem,
                "genre"       : genre,
                "audio_path"  : audio_map[stem],
                "lyrics_path" : lyrics_map[stem],
            })

    df = pd.DataFrame(records)
    print(f"[data_loader] Manifest built: {len(df)} paired songs across {df['genre'].nunique()} genres")
    print(df["genre"].value_counts().to_string())
    return df


if __name__ == "__main__":
    df = build_manifest()
    print(df.head())
