"""
Configuration file for the Lyric + Audio Genre Fusion Classifier.
All paths, hyperparameters, and settings are centralized here.
"""

import os

# ─── Dataset Paths ────────────────────────────────────────────────────────────
DATASET_ZIP     = "Audio_Lyrics_Dataset.zip"       # Put your zip file here
DATASET_ROOT    = "Audio_Lyrics_Dataset"
AUDIO_DIR       = os.path.join(DATASET_ROOT, "Audio")
LYRICS_DIR      = os.path.join(DATASET_ROOT, "Lyrics")

# Structured local datasets
DATASETS_DIR         = "datasets"
MAIN_DATASET_ROOT    = os.path.join(DATASETS_DIR, "Audio_Lyrics_Dataset")
CROSS_DATASET_ROOT   = os.path.join(DATASETS_DIR, "CrossDataset")
# Use all available paired songs per genre (no cap).
# Set to an int (e.g. 50) to limit dataset size for faster experiments.
TRAIN_SONGS_PER_GENRE = None

# ─── Genre Labels ─────────────────────────────────────────────────────────────
GENRES = [
    "BLUES",
    "ELECTRONIC - EDM",
    "HIP-HOP - RAP",
    "JAZZ",
    "METAL",
    "POP",
    "R&B - SOUL",
    "ROCK",
]
NUM_CLASSES = len(GENRES)

# ─── Audio Feature Extraction ─────────────────────────────────────────────────
SAMPLE_RATE     = 22050     # librosa default
DURATION        = 30        # seconds to load per clip (first 30s)
N_MFCC          = 40        # number of MFCC coefficients
N_FFT           = 2048
HOP_LENGTH      = 512
N_MELS          = 128
N_CHROMA        = 12
N_CONTRAST      = 7         # spectral contrast bands

# ─── Lyric / Text Features ────────────────────────────────────────────────────
TFIDF_MAX_FEATURES  = 5000
LYRIC_EMBED_MODEL   = "sentence-transformers/all-mpnet-base-v2"
LYRIC_EMBED_DIM     = 768

# ─── Model / Training ─────────────────────────────────────────────────────────
TEST_SIZE       = 0.20
VAL_SIZE        = 0.10      # fraction of training set used for validation
RANDOM_STATE    = 42

# MLP hyperparameters
MLP_HIDDEN_LAYERS   = (512, 256, 128)
MLP_DROPOUT         = 0.3
MLP_LR              = 1e-3
MLP_EPOCHS          = 100
MLP_BATCH_SIZE      = 32
MLP_PATIENCE        = 15    # early stopping patience
MLP_LABEL_SMOOTHING = 0.05

# Fusion / multimodal settings
PROJ_DIM            = 256
FUSION_HIDDEN       = (512, 256, 128)
LATE_FUSION_CV      = 3

# CNN audio model on mel-spectrogram
MEL_IMG_FRAMES      = 256
CNN_EPOCHS          = 40
CNN_BATCH_SIZE      = 16
CNN_LR              = 5e-4
CNN_PATIENCE        = 8

# ─── Output / Artefacts ───────────────────────────────────────────────────────
OUTPUT_DIR      = "outputs"
FEATURE_CACHE   = os.path.join(OUTPUT_DIR, "features_cache.pkl")
RESULTS_DIR     = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR      = os.path.join(OUTPUT_DIR, "models")

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)
