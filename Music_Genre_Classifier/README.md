# Lyric + Audio Genre Fusion Classifier
### Graduation Project — Music Information Retrieval

---

## Project Structure

```
genre_classifier/
├── config.py              # All paths & hyperparameters
├── data_loader.py         # Dataset manifest builder
├── audio_features.py      # MFCC, spectral contrast, tempo, chroma, mel
├── lyric_features.py      # TF-IDF & sentence-transformer embeddings
├── feature_pipeline.py    # Orchestration + feature caching
├── models.py              # Audio MLP, Lyric MLP, Early Fusion, Late Fusion
├── evaluation.py          # Metrics, confusion matrices, charts
├── train.py               # Main experiment runner (all 5 models)
├── predict.py             # Inference CLI for new songs
└── requirements.txt
```

---

## Dataset Layout Expected

```
Audio_Lyrics_Dataset/
├── Audio/
│   ├── BLUES/          *.mp3
│   ├── ELECTRONIC - EDM/
│   ├── HIP-HOP - RAP/
│   ├── JAZZ/
│   ├── METAL/
│   ├── POP/
│   ├── R&B - SOUL/
│   └── ROCK/
└── Lyrics/
    ├── BLUES/          *.txt
    ├── ...             (same structure)
```

Place `Audio_Lyrics_Dataset.zip` in the same directory as the scripts.
The code auto-extracts on first run.

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) GPU support — replace torch with CUDA build:
#    pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Training

```bash
# Run all experiments (extracts features, trains 5 models, saves results)
python train.py

# Force re-extraction of features (e.g. after changing feature config)
python train.py --rebuild
```

**What `train.py` does:**

| Step | Model | Modality |
|------|-------|----------|
| 3 | Unimodal Audio MLP | Audio only |
| 4 | Unimodal Lyric Embedding MLP | Lyrics only (sentence-transformers) |
| 5 | Unimodal TF-IDF + SVM | Lyrics only (bag-of-words) |
| 6 | **Early Fusion MLP** | Audio ‖ Lyric concatenation → single MLP |
| 7 | **Late Fusion** | Separate audio + lyric MLPs → avg softmax |
| 9 | 5-Fold CV | Early Fusion cross-validation |

---

## Outputs

All results are saved to `outputs/`:

```
outputs/
├── features_cache.pkl           # Cached features (skip re-extraction)
├── models/
│   ├── audio_mlp.pkl
│   ├── lyric_mlp.pkl
│   ├── tfidf_svm.pkl
│   ├── early_fusion_mlp.pkl
│   └── late_fusion.pkl
└── results/
    ├── cm_unimodal_audio_mlp.png
    ├── cm_early_fusion_mlp.png
    ├── ...
    ├── model_comparison.png     # Bar chart of all models
    └── results_summary.json
```

---

## Inference

```bash
# Multimodal prediction (most accurate)
python predict.py --audio song.mp3 --lyrics song.txt

# Audio only
python predict.py --audio song.mp3

# Lyrics from text file
python predict.py --lyrics song.txt

# Raw lyrics string
python predict.py --text "I used to roll the dice, fear the devil's eyes..."

# Show top 5 predictions
python predict.py --audio song.mp3 --lyrics song.txt --top 5
```

**Example output:**
```
  Predicted genre : ROCK
  Confidence      : 87.3 %

  Top-3 predictions:
    1. ROCK                87.3 %  ██████████████████████████
    2. METAL                8.1 %  ██
    3. BLUES                2.4 %
```

---

## Audio Features Extracted

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| MFCC (mean+std) | 80 | Timbral texture |
| Delta MFCC | 80 | MFCC temporal derivatives |
| Chroma STFT | 24 | Harmonic / pitch class content |
| Spectral Contrast | 14 | Tonal vs noise components |
| Mel-Spectrogram | 256 | Perceptual energy distribution |
| Spectral Centroid | 2 | Brightness |
| Spectral Rolloff | 2 | High-frequency content |
| Zero-Crossing Rate | 2 | Noisiness / percussiveness |
| Tempo | 1 | BPM |
| **Total** | **~461** | |

## Lyric Features

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| TF-IDF (1-2 gram) | 5000 | Sparse BoW (SVM baseline) |
| Sentence Embedding | 384 | MiniLM-L6-v2 semantic vectors |

---

## Evaluation Metrics

- **Accuracy** — overall classification rate
- **F1-Macro** — unweighted average F1 across all 8 genres
- **F1-Weighted** — class-frequency weighted F1
- **Confusion Matrix** — per-class error patterns
- **5-Fold Cross-Validation** — generalisation estimate

---

## Key Design Decisions

**Early Fusion** concatenates audio and lyric feature vectors before feeding
into a single MLP. This allows the network to learn cross-modal interactions
(e.g. "fast tempo + aggressive lyrics → Metal").

**Late Fusion** trains two completely independent models and averages their
softmax probability outputs. This is more robust when one modality is missing
or noisy, since each model specialises on its domain.

**Sentence transformers** are used for lyric embeddings instead of raw TF-IDF
in the fusion models because they capture semantic meaning ("violent imagery"
in Metal vs Hip-Hop) rather than just word frequency.

---

## Genres

`BLUES` · `ELECTRONIC - EDM` · `HIP-HOP - RAP` · `JAZZ` · `METAL` · `POP` · `R&B - SOUL` · `ROCK`

50 songs × 8 genres = **400 paired samples**
