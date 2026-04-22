"""
train.py
────────
Main experiment runner.

Runs the full unimodal vs multimodal comparison study:

  Step 1  –  Extract / load cached features
  Step 2  –  Train–val–test split
  Step 3  –  Unimodal Audio MLP
  Step 4  –  Unimodal Lyric Embedding MLP
  Step 5  –  Unimodal TF-IDF + SVM (text baseline)
  Step 6  –  Early Fusion MLP  (audio + lyric concat)
  Step 7  –  Late Fusion  (separate audio + lyric models, avg probs)
  Step 8  –  Print comparison table + save charts
  Step 9  –  Stratified k-fold cross-validation on Early Fusion

Usage:
    python train.py
    python train.py --rebuild      # force re-extract features
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm

import config
from feature_pipeline   import build_or_load_features
from data_loader        import build_manifest
from audio_features     import extract_audio_features, extract_mel_spectrogram_image
from lyric_features     import load_lyrics, embed_lyrics_batch
from models             import (
    build_audio_classifier,
    build_lyric_mlp_classifier,
    build_cnn_audio_classifier,
    build_tfidf_svm_pipeline,
    build_early_fusion_classifier,
    LearnableLateFusionClassifier,
)
from evaluation import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_training_curves,
    compare_models,
    cross_val_evaluate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _split_train_val(X, y, val_size=config.VAL_SIZE):
    """Returns X_train, X_val, y_train, y_val."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_val, y_train, y_val


def _split_indices(y, val_size=config.VAL_SIZE):
    """Single stratified split indices (keeps multimodal alignment)."""
    idx_all = np.arange(len(y))
    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=val_size,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    return idx_train, idx_val


def save_model(obj, name: str):
    path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [train] Model saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(rebuild: bool = False):

    # ── 1. Features ────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 1 — Load / Extract Features")
    print("═"*60)
    data  = build_or_load_features(force_rebuild=rebuild)

    X_audio  = data["audio_features"]     # (N, D_a)
    X_mel    = data["mel_images"]         # (N, 1, N_MELS, T)
    X_lyric  = data["lyric_embeddings"]   # (N, D_l)
    texts    = data["lyric_texts"]
    y        = data["labels"]
    le       = data["label_encoder"]

    print(f"\n  Audio  feature dim : {X_audio.shape[1]}")
    print(f"  Mel image shape    : {X_mel.shape[1:]}")
    print(f"  Lyric  feature dim : {X_lyric.shape[1]}")
    print(f"  Classes            : {list(le.classes_)}")

    # ── 2. Splits ──────────────────────────────────────────────────────────────
    # IMPORTANT: one index split for ALL modalities (keeps paired samples aligned)
    idx_train, idx_val = _split_indices(y, val_size=config.VAL_SIZE)
    y_tr, y_val = y[idx_train], y[idx_val]

    X_audio_tr, X_audio_val = X_audio[idx_train], X_audio[idx_val]
    X_lyric_tr, X_lyric_val = X_lyric[idx_train], X_lyric[idx_val]
    X_mel_tr,   X_mel_val   = X_mel[idx_train],   X_mel[idx_val]

    texts_arr = np.array(texts)
    texts_tr  = texts_arr[idx_train].tolist()
    texts_val = texts_arr[idx_val].tolist()
    # Test will come ONLY from cross-dataset
    texts_te  = []
    y_tfidf_tr  = y[idx_train]
    y_tfidf_val = y[idx_val]
    y_tfidf_te  = np.array([], dtype=np.int64)

    # ── Cross-dataset test set (external) ─────────────────────────────────────
    print("\n  Building cross-dataset test set …")
    df_cross = build_manifest(dataset_root=config.CROSS_DATASET_ROOT, max_per_genre=None)
    # keep only known genres
    df_cross = df_cross[df_cross["genre"].isin(le.classes_)].reset_index(drop=True)
    if len(df_cross) == 0:
        raise RuntimeError("CrossDataset manifest is empty after filtering known genres.")

    X_audio_te = []
    X_mel_te = []
    lyric_texts_te = []
    for ap, lp in tqdm(zip(df_cross["audio_path"], df_cross["lyrics_path"]), total=len(df_cross), desc="Cross-Test Features", unit="song"):
        X_audio_te.append(extract_audio_features(ap))
        X_mel_te.append(extract_mel_spectrogram_image(ap))
        lyric_texts_te.append(load_lyrics(lp))
    X_audio_te = np.vstack(X_audio_te).astype(np.float32)
    X_mel_te = np.stack(X_mel_te).astype(np.float32)
    X_lyric_te = embed_lyrics_batch(lyric_texts_te, show_progress=True)
    y_te = le.transform(df_cross["genre"].values)
    texts_te = lyric_texts_te
    y_tfidf_te = y_te.copy()

    print(f"\n  Train : {len(y_tr)}  |  Val : {len(y_val)}  |  Cross-Test : {len(y_te)}")

    all_results = []

    # ── 3. Unimodal Audio MLP ──────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 3 — Unimodal Audio MLP")
    print("═"*60)
    audio_mlp = build_audio_classifier(X_audio.shape[1])
    audio_mlp.fit(X_audio_tr, y_tr, X_audio_val, y_val)
    y_pred_audio = audio_mlp.predict(X_audio_te)
    res_audio = evaluate_predictions(y_te, y_pred_audio, le, "Unimodal Audio MLP")
    plot_confusion_matrix(y_te, y_pred_audio, le, "Unimodal Audio MLP")
    plot_training_curves(audio_mlp.train_losses_, audio_mlp.val_losses_, "Unimodal Audio MLP")
    save_model(audio_mlp, "audio_mlp")
    all_results.append(res_audio)

    # ── 3b. CNN Audio on Mel-Spectrogram ──────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 3b — CNN Audio (Mel-Spectrogram)")
    print("═"*60)
    cnn_audio = build_cnn_audio_classifier()
    cnn_audio.fit(X_mel_tr, y_tr, X_mel_val, y_val)
    y_pred_cnn = cnn_audio.predict(X_mel_te)
    res_cnn = evaluate_predictions(y_te, y_pred_cnn, le, "Unimodal CNN Audio (Mel)")
    plot_confusion_matrix(y_te, y_pred_cnn, le, "Unimodal CNN Audio (Mel)")
    save_model(cnn_audio, "cnn_audio_mel")
    all_results.append(res_cnn)

    # ── 4. Unimodal Lyric Embedding MLP ───────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 4 — Unimodal Lyric Embedding MLP")
    print("═"*60)
    lyric_mlp = build_lyric_mlp_classifier(X_lyric.shape[1])
    lyric_mlp.fit(X_lyric_tr, y_tr, X_lyric_val, y_val)
    y_pred_lyric = lyric_mlp.predict(X_lyric_te)
    res_lyric = evaluate_predictions(y_te, y_pred_lyric, le, "Unimodal Lyric Embedding MLP")
    plot_confusion_matrix(y_te, y_pred_lyric, le, "Unimodal Lyric Embedding MLP")
    plot_training_curves(lyric_mlp.train_losses_, lyric_mlp.val_losses_, "Unimodal Lyric Embedding MLP")
    save_model(lyric_mlp, "lyric_mlp")
    all_results.append(res_lyric)

    # ── 5. Unimodal TF-IDF + SVM ──────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 5 — Unimodal TF-IDF + SVM")
    print("═"*60)
    tfidf_svm = build_tfidf_svm_pipeline()
    tfidf_svm.fit(texts_tr, y_tfidf_tr)    # sklearn pipeline, no val needed
    y_pred_svm = tfidf_svm.predict(texts_te)
    res_svm = evaluate_predictions(y_tfidf_te, y_pred_svm, le, "Unimodal TF-IDF + SVM")
    plot_confusion_matrix(y_tfidf_te, y_pred_svm, le, "Unimodal TF-IDF + SVM")
    save_model(tfidf_svm, "tfidf_svm")
    all_results.append(res_svm)

    # ── 6. Early Fusion MLP ────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 6 — Early Fusion MLP  (audio ‖ lyric)")
    print("═"*60)
    X_ef_tr  = np.concatenate([X_audio_tr,  X_lyric_tr],  axis=1)
    X_ef_val = np.concatenate([X_audio_val, X_lyric_val], axis=1)
    X_ef_te  = np.concatenate([X_audio_te,  X_lyric_te],  axis=1)

    ef_candidates = [7, 13, 21, 42, 77]
    best_ef = None
    best_ef_val = -1.0
    for seed in ef_candidates:
        ef_model = build_early_fusion_classifier(X_audio.shape[1], X_lyric.shape[1], seed=seed)
        ef_model.fit(X_audio_tr, X_lyric_tr, y_tr, X_audio_val, X_lyric_val, y_val)
        ef_val_pred = ef_model.predict(X_audio_val, X_lyric_val)
        ef_val_acc = float(np.mean(ef_val_pred == y_val))
        print(f"  [EarlyFusion-Gated] seed={seed} val_acc={ef_val_acc:.4f}")
        if ef_val_acc > best_ef_val:
            best_ef_val = ef_val_acc
            best_ef = ef_model

    ef_mlp = best_ef
    print(f"  [EarlyFusion-Gated] selected val_acc={best_ef_val:.4f}")
    y_pred_ef = ef_mlp.predict(X_audio_te, X_lyric_te)
    res_ef = evaluate_predictions(y_te, y_pred_ef, le, "Early Fusion (Gated MLP)")
    plot_confusion_matrix(y_te, y_pred_ef, le, "Early Fusion (Gated MLP)")
    plot_training_curves(ef_mlp.base_.train_losses_, ef_mlp.base_.val_losses_, "Early Fusion (Gated MLP)")
    save_model(ef_mlp, "early_fusion_gated_mlp")
    all_results.append(res_ef)

    # ── 6b. Early Fusion (Classical ML on fused vectors) ──────────────────────
    print("\n" + "═"*60)
    print("  STEP 6b — Classical Fusion (SVC / LogReg / ExtraTrees)")
    print("═"*60)

    X_fused_tr  = np.concatenate([X_audio_tr,  X_lyric_tr],  axis=1)
    X_fused_val = np.concatenate([X_audio_val, X_lyric_val], axis=1)
    X_fused_te  = np.concatenate([X_audio_te,  X_lyric_te],  axis=1)

    def _val_acc(clf):
        return float(np.mean(clf.predict(X_fused_val) == y_val))

    classical_candidates = []

    # (1) RBF-SVC (needs scaling). Use calibrated wrapper to get usable confidences.
    svc_base = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=config.RANDOM_STATE)),
    ])
    svc_param = {
        "svc__C": np.logspace(-1, 2, 12),
        "svc__gamma": ["scale", "auto"],
    }
    svc_search = RandomizedSearchCV(
        estimator=svc_base,
        param_distributions=svc_param,
        n_iter=16,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    svc_search.fit(X_fused_tr, y_tr)
    svc_best = svc_search.best_estimator_
    svc_cal = CalibratedClassifierCV(svc_best, method="sigmoid", cv=3)
    svc_cal.fit(X_fused_tr, y_tr)
    classical_candidates.append(("Fusion SVC (RBF, calibrated)", svc_cal))

    # (2) Multinomial Logistic Regression baseline (strong, fast).
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )),
    ])
    lr.fit(X_fused_tr, y_tr)
    classical_candidates.append(("Fusion LogisticRegression", lr))

    # (3) ExtraTrees (nonlinear, no scaling needed).
    et = ExtraTreesClassifier(
        n_estimators=800,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
    )
    et.fit(X_fused_tr, y_tr)
    classical_candidates.append(("Fusion ExtraTrees", et))

    best_name, best_clf, best_val_acc = None, None, -1.0
    for name, clf in classical_candidates:
        acc = _val_acc(clf)
        print(f"  [{name}] val_acc={acc:.4f}")
        if acc > best_val_acc:
            best_val_acc = acc
            best_name, best_clf = name, clf

    print(f"  [ClassicalFusion] selected: {best_name} (val_acc={best_val_acc:.4f})")
    y_pred_cf = best_clf.predict(X_fused_te)
    res_cf = evaluate_predictions(y_te, y_pred_cf, le, best_name)
    plot_confusion_matrix(y_te, y_pred_cf, le, best_name)
    save_model(best_clf, "classical_fusion_best")
    all_results.append(res_cf)

    # ── 7. Late Fusion ─────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 7 — Late Fusion  (avg softmax)")
    print("═"*60)
    lf_candidates = [7, 13, 21, 42, 77]
    best_lf = None
    best_lf_val = -1.0
    for seed in lf_candidates:
        lf_model = LearnableLateFusionClassifier(X_audio.shape[1], X_lyric.shape[1], seed=seed)
        lf_model.fit(X_audio_tr, X_lyric_tr, y_tr, X_audio_val, X_lyric_val, y_val)
        lf_val_pred = lf_model.predict(X_audio_val, X_lyric_val)
        lf_val_acc = float(np.mean(lf_val_pred == y_val))
        print(f"  [LateFusion-Learnable] seed={seed} val_acc={lf_val_acc:.4f}")
        if lf_val_acc > best_lf_val:
            best_lf_val = lf_val_acc
            best_lf = lf_model

    lf_clf = best_lf
    print(f"  [LateFusion-Learnable] selected val_acc={best_lf_val:.4f}")
    y_pred_lf = lf_clf.predict(X_audio_te, X_lyric_te)
    res_lf = evaluate_predictions(y_te, y_pred_lf, le, "Late Fusion (Learnable)")
    plot_confusion_matrix(y_te, y_pred_lf, le, "Late Fusion (Learnable)")
    save_model(lf_clf, "late_fusion_learnable")
    all_results.append(res_lf)

    # ── 8. Comparison ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 8 — Model Comparison")
    print("═"*60)
    compare_models(all_results)

    # ── 9. Cross-validation on best model (Early Fusion) ──────────────────────
    print("\n" + "═"*60)
    print("  STEP 9 — 5-Fold CV on Fusion Models")
    print("═"*60)
    X_early_fusion = np.concatenate([X_audio, X_lyric], axis=1)

    def ef_factory(X_tr, y_tr):
        # Lightweight fallback for CV speed: plain MLP on fused vector
        clf = build_audio_classifier(X_early_fusion.shape[1])
        clf.fit(X_tr, y_tr)
        return clf

    cross_val_evaluate(ef_factory, X_early_fusion, y,
                       k=5, model_name="Fusion MLP (vector)")

    print("\n[train] All experiments complete.")
    print(f"  Outputs saved to: {config.OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genre Fusion Classifier — Training")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force re-extraction of all features")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
