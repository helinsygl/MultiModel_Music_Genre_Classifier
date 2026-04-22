"""
evaluation.py
─────────────
All evaluation metrics and reporting utilities.

Functions:
  evaluate_predictions()      → dict with accuracy, f1_macro, f1_weighted, per-class report
  plot_confusion_matrix()     → saves a heatmap PNG
  plot_training_curves()      → saves a loss-curve PNG
  compare_models()            → prints a summary table
  cross_val_evaluate()        → stratified k-fold evaluation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import config


# ─── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          label_encoder: LabelEncoder,
                          model_name: str = "Model") -> dict:
    """
    Compute accuracy, macro-F1, weighted-F1, and per-class report.
    Returns a results dict and prints a formatted summary.
    """
    acc       = accuracy_score(y_true, y_pred)
    bal_acc   = balanced_accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = (cm.diagonal() / np.clip(cm.sum(axis=1), 1, None)).tolist()

    class_names = label_encoder.classes_
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    results = {
        "model"       : model_name,
        "accuracy"    : round(acc,        4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro"    : round(f1_macro,   4),
        "f1_weighted" : round(f1_weighted,4),
        "per_class_accuracy": [round(float(x), 4) for x in per_class_acc],
        "report"      : report,
    }

    print(f"\n{'─'*60}")
    print(f"  {model_name}")
    print(f"{'─'*60}")
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  F1 (macro)        : {f1_macro:.4f}")
    print(f"  F1 (weighted)     : {f1_weighted:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return results


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           label_encoder: LabelEncoder,
                           model_name: str = "Model",
                           save_dir: str = config.RESULTS_DIR) -> str:
    """Plot and save a normalised confusion matrix. Returns the file path."""
    class_names  = label_encoder.classes_
    cm           = confusion_matrix(y_true, y_pred)
    cm_norm      = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, title, fmt in zip(
            axes,
            [cm,       cm_norm],
            ["Counts", "Normalised"],
            ["d",      ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(f"{model_name} — Confusion Matrix ({title})", fontsize=13)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_").replace("/", "-")
    path = os.path.join(save_dir, f"cm_{slug}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [evaluation] Confusion matrix saved → {path}")
    return path


# ─── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(train_losses: list,
                          val_losses: list,
                          model_name: str = "Model",
                          save_dir: str = config.RESULTS_DIR) -> str:
    """Plot train / val loss curves and save as PNG."""
    if not train_losses:
        return ""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    if val_losses:
        plt.plot(val_losses, label="Val Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{model_name} — Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(save_dir, f"curves_{slug}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [evaluation] Training curves saved → {path}")
    return path


# ─── Model comparison table ───────────────────────────────────────────────────

def compare_models(results_list: list,
                   save_dir: str = config.RESULTS_DIR) -> pd.DataFrame:
    """
    Build and print a comparison table from a list of results dicts.
    Saves a PNG bar chart and returns a DataFrame.
    """
    rows = [(r["model"], r["accuracy"], r["f1_macro"], r["f1_weighted"])
            for r in results_list]
    df = pd.DataFrame(rows, columns=["Model", "Accuracy", "F1-Macro", "F1-Weighted"])
    df = df.sort_values("F1-Macro", ascending=False).reset_index(drop=True)

    print("\n" + "═"*60)
    print("  MODEL COMPARISON SUMMARY")
    print("═"*60)
    print(df.to_string(index=False))
    print("═"*60 + "\n")

    # Bar chart
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, label) in enumerate([
            ("Accuracy",    "Accuracy"),
            ("F1-Macro",    "F1-Macro"),
            ("F1-Weighted", "F1-Weighted"),
    ]):
        ax.bar(x + i*width, df[col], width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Unimodal vs Multimodal Genre Classification")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [evaluation] Comparison chart saved → {path}")

    # Save JSON
    json_path = os.path.join(save_dir, "results_summary.json")
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "report"} for r in results_list],
                  f, indent=2)
    return df


# ─── Stratified k-fold cross-validation ──────────────────────────────────────

def cross_val_evaluate(model_fn,
                        X: np.ndarray,
                        y: np.ndarray,
                        k: int = 5,
                        model_name: str = "Model") -> dict:
    """
    Stratified k-fold CV for a model factory function.
    model_fn(X_train, y_train) must return a fitted object with .predict(X).
    Returns mean and std of accuracy and F1-macro.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)
    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = model_fn(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        print(f"    Fold {fold}/{k}: acc={acc:.4f}  f1={f1:.4f}")

    result = {
        "model"        : model_name,
        "cv_acc_mean"  : np.mean(accs),
        "cv_acc_std"   : np.std(accs),
        "cv_f1_mean"   : np.mean(f1s),
        "cv_f1_std"    : np.std(f1s),
    }
    print(f"  {model_name}: acc={result['cv_acc_mean']:.4f}±{result['cv_acc_std']:.4f}  "
          f"f1={result['cv_f1_mean']:.4f}±{result['cv_f1_std']:.4f}")
    return result
