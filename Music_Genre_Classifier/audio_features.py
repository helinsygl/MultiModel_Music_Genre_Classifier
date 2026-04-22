"""
audio_features.py
─────────────────
Extract acoustic features from .mp3 files using librosa.

Features extracted per song (all summarised as mean + std over time):
  • MFCC (40 coefficients)            →  80 dims
  • Delta MFCC                        →  80 dims
  • Chroma STFT (12 bins)             →  24 dims
  • Spectral contrast (7 bands)       →  14 dims
  • Mel-spectrogram (128 bins)        → 256 dims
  • Spectral centroid                 →   2 dims
  • Spectral rolloff                  →   2 dims
  • Zero-crossing rate                →   2 dims
  • Tempo (single scalar)             →   1 dim
  ──────────────────────────────────────────────
  Total audio feature vector         → 461 dims
"""

import os
import numpy as np
import librosa
from librosa.util import fix_length
import warnings
warnings.filterwarnings("ignore")

import config


def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Load an audio file and return a 1-D feature vector.
    Returns a zero vector of the correct length on failure.
    """
    try:
        y, sr = librosa.load(
            audio_path,
            sr=config.SAMPLE_RATE,
            duration=config.DURATION,
            mono=True,
        )
    except Exception as e:
        print(f"[audio_features] ERROR loading {audio_path}: {e}")
        return np.zeros(_feature_dim(), dtype=np.float32)

    feats = []

    # ── MFCC ──────────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    feats.extend(_stats(mfcc))   # 80 dims

    # ── Delta MFCC ────────────────────────────────────────────────────────────
    delta_mfcc = librosa.feature.delta(mfcc)
    feats.extend(_stats(delta_mfcc))   # 80 dims

    # ── Chroma STFT ───────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr,
        n_chroma=config.N_CHROMA,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    feats.extend(_stats(chroma))   # 24 dims

    # ── Spectral Contrast ─────────────────────────────────────────────────────
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr,
        n_bands=config.N_CONTRAST,
        # Must keep octave bands within Nyquist (sr/2) for low sample rates
        fmin=40.0,
        hop_length=config.HOP_LENGTH,
    )
    feats.extend(_stats(contrast))   # 14 dims

    # ── Mel-spectrogram ───────────────────────────────────────────────────────
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats.extend(_stats(mel_db))   # 256 dims

    # ── Spectral Centroid ─────────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    feats.extend(_stats(centroid))   # 2 dims

    # ── Spectral Rolloff ──────────────────────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    feats.extend(_stats(rolloff))   # 2 dims

    # ── Zero-Crossing Rate ────────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=config.HOP_LENGTH)
    feats.extend(_stats(zcr))   # 2 dims

    # ── Tempo ─────────────────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats.append(float(tempo) if np.isscalar(tempo) else float(tempo[0]))   # 1 dim

    return np.array(feats, dtype=np.float32)


def extract_mel_spectrogram_image(audio_path: str) -> np.ndarray:
    """
    Return a fixed-size log-mel image for CNN input: (1, N_MELS, MEL_IMG_FRAMES).
    """
    try:
        y, sr = librosa.load(
            audio_path,
            sr=config.SAMPLE_RATE,
            duration=config.DURATION,
            mono=True,
        )
    except Exception as e:
        print(f"[audio_features] ERROR loading for mel image {audio_path}: {e}")
        return np.zeros((1, config.N_MELS, config.MEL_IMG_FRAMES), dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = fix_length(mel_db, size=config.MEL_IMG_FRAMES, axis=1)

    # Per-sample normalisation for stable CNN training
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
    return mel_db[np.newaxis, :, :].astype(np.float32)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _stats(matrix: np.ndarray):
    """Return [mean_over_time, std_over_time] for each row."""
    return list(np.mean(matrix, axis=1)) + list(np.std(matrix, axis=1))


def _feature_dim() -> int:
    """Return total number of audio features (hard-coded for speed)."""
    return (
        2 * config.N_MFCC          # MFCC
        + 2 * config.N_MFCC        # delta MFCC
        + 2 * config.N_CHROMA      # chroma
        + 2 * (config.N_CONTRAST + 1)  # spectral contrast (n_bands+1 rows)
        + 2 * config.N_MELS        # mel
        + 2                        # centroid
        + 2                        # rolloff
        + 2                        # ZCR
        + 1                        # tempo
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        vec = extract_audio_features(sys.argv[1])
        print(f"Feature vector shape: {vec.shape}")
        print(vec[:10])
