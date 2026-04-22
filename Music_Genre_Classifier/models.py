"""
Model zoo for multimodal music genre classification.
"""

import random
from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], n_classes: int, dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-like wrapper around a PyTorch MLP."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int = config.NUM_CLASSES,
        hidden: Tuple[int, ...] = config.MLP_HIDDEN_LAYERS,
        dropout: float = config.MLP_DROPOUT,
        lr: float = config.MLP_LR,
        epochs: int = config.MLP_EPOCHS,
        batch_size: int = config.MLP_BATCH_SIZE,
        patience: int = config.MLP_PATIENCE,
        device: Optional[str] = None,
        seed: int = config.RANDOM_STATE,
    ):
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        self.model_ = None
        self.scaler_ = StandardScaler()
        self.classes_ = np.arange(n_classes)
        self.train_losses_ = []
        self.val_losses_ = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        _set_seed(self.seed)
        X_train = self.scaler_.fit_transform(X_train).astype(np.float32)
        if X_val is not None:
            X_val = self.scaler_.transform(X_val).astype(np.float32)

        Xt = torch.tensor(X_train)
        yt = torch.tensor(y_train, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        if X_val is not None:
            Xv = torch.tensor(X_val).to(self.device)
            yv = torch.tensor(y_val, dtype=torch.long).to(self.device)

        self.model_ = _MLP(self.in_dim, self.hidden, self.n_classes, self.dropout).to(self.device)
        optimiser = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.MLP_LABEL_SMOOTHING)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        best_val_loss = float("inf")
        best_state = None
        patience_cnt = 0

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    loss = criterion(self.model_(xb), yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(X_train)
            self.train_losses_.append(epoch_loss)
            scheduler.step()

            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model_(Xv), yv).item()
                self.val_losses_.append(val_loss)
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_cnt = 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        print(f"    [EarlyStopping] epoch {epoch}/{self.epochs}, val_loss={val_loss:.4f}")
                        break
                if epoch % 10 == 0:
                    print(f"    Epoch {epoch:3d}/{self.epochs} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f}")
            elif epoch % 10 == 0:
                print(f"    Epoch {epoch:3d}/{self.epochs} | train_loss={epoch_loss:.4f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        X = self.scaler_.transform(X).astype(np.float32)
        Xt = torch.tensor(X).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model_(Xt), dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class GatedEarlyFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Early fusion with modality projections and learnable gating.
    """

    def __init__(self, audio_dim: int, lyric_dim: int, seed: int = config.RANDOM_STATE):
        self.audio_dim = audio_dim
        self.lyric_dim = lyric_dim
        self.seed = seed
        in_dim = 2 * config.PROJ_DIM
        self.audio_scaler_ = StandardScaler()
        self.lyric_scaler_ = StandardScaler()
        self.base_ = MLPClassifier(
            in_dim=in_dim,
            hidden=config.FUSION_HIDDEN,
            seed=seed,
        )
        self.audio_proj_ = None
        self.lyric_proj_ = None
        self.gate_ = None
        self.classes_ = np.arange(config.NUM_CLASSES)

    def _build_heads(self):
        self.audio_proj_ = torch.nn.Linear(self.audio_dim, config.PROJ_DIM)
        self.lyric_proj_ = torch.nn.Linear(self.lyric_dim, config.PROJ_DIM)
        self.gate_ = torch.nn.Sequential(
            torch.nn.Linear(2 * config.PROJ_DIM, config.PROJ_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(config.PROJ_DIM, config.PROJ_DIM),
            torch.nn.Sigmoid(),
        )

    def _fuse_np(self, Xa, Xl):
        if self.audio_proj_ is None:
            _set_seed(self.seed)
            self._build_heads()
        with torch.no_grad():
            ta = torch.tensor(Xa, dtype=torch.float32)
            tl = torch.tensor(Xl, dtype=torch.float32)
            pa = self.audio_proj_(ta)
            pl = self.lyric_proj_(tl)
            g = self.gate_(torch.cat([pa, pl], dim=1))
            fused = torch.cat([g * pa, (1.0 - g) * pl], dim=1)
        return fused.numpy().astype(np.float32)

    def fit(self, X_audio, X_lyric, y, X_audio_val=None, X_lyric_val=None, y_val=None):
        Xa = self.audio_scaler_.fit_transform(X_audio).astype(np.float32)
        Xl = self.lyric_scaler_.fit_transform(X_lyric).astype(np.float32)
        Xf = self._fuse_np(Xa, Xl)

        if X_audio_val is not None:
            Xav = self.audio_scaler_.transform(X_audio_val).astype(np.float32)
            Xlv = self.lyric_scaler_.transform(X_lyric_val).astype(np.float32)
            Xfv = self._fuse_np(Xav, Xlv)
        else:
            Xfv = None

        self.base_.fit(Xf, y, Xfv, y_val)
        return self

    def predict_proba(self, X_audio, X_lyric):
        Xa = self.audio_scaler_.transform(X_audio).astype(np.float32)
        Xl = self.lyric_scaler_.transform(X_lyric).astype(np.float32)
        Xf = self._fuse_np(Xa, Xl)
        return self.base_.predict_proba(Xf)

    def predict(self, X_audio, X_lyric):
        return np.argmax(self.predict_proba(X_audio, X_lyric), axis=1)


class _MelCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.head(h)


class CNNAudioClassifier(BaseEstimator, ClassifierMixin):
    """CNN classifier on mel-spectrogram images."""

    def __init__(self, n_classes: int = config.NUM_CLASSES, seed: int = config.RANDOM_STATE, device: Optional[str] = None):
        self.n_classes = n_classes
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.classes_ = np.arange(n_classes)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        _set_seed(self.seed)
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=config.CNN_BATCH_SIZE, shuffle=True)

        if X_val is not None:
            Xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            yv = torch.tensor(y_val, dtype=torch.long).to(self.device)

        self.model_ = _MelCNN(self.n_classes).to(self.device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=config.CNN_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.CNN_EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.MLP_LABEL_SMOOTHING)

        best_state = None
        best_val = float("inf")
        patience = 0

        for epoch in range(1, config.CNN_EPOCHS + 1):
            self.model_.train()
            run_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(xb)
            scheduler.step()

            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    vl = criterion(self.model_(Xv), yv).item()
                if vl < best_val - 1e-4:
                    best_val = vl
                    patience = 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience += 1
                    if patience >= config.CNN_PATIENCE:
                        break
            if epoch % 10 == 0:
                print(f"    [CNN] epoch={epoch} train_loss={run_loss/len(X_train):.4f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(Xt), dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build_audio_classifier(X_dim: int, seed: Optional[int] = None) -> MLPClassifier:
    return MLPClassifier(in_dim=X_dim, seed=config.RANDOM_STATE if seed is None else seed)


def build_lyric_mlp_classifier(X_dim: int, seed: Optional[int] = None) -> MLPClassifier:
    return MLPClassifier(
        in_dim=X_dim,
        hidden=(256, 128),
        seed=config.RANDOM_STATE if seed is None else seed,
    )


def build_tfidf_svm_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("svm", CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"),
            method="sigmoid",
            cv=config.LATE_FUSION_CV,
        )),
    ])


def build_early_fusion_classifier(audio_dim: int, lyric_dim: int, seed: Optional[int] = None) -> GatedEarlyFusionClassifier:
    return GatedEarlyFusionClassifier(audio_dim, lyric_dim, seed=config.RANDOM_STATE if seed is None else seed)


def build_cnn_audio_classifier() -> CNNAudioClassifier:
    return CNNAudioClassifier()


class LearnableLateFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Trains audio + lyric experts then learns fusion on validation probabilities.
    """

    def __init__(self, audio_dim: int, lyric_dim: int, seed: int = config.RANDOM_STATE):
        self.audio_clf_ = build_audio_classifier(audio_dim, seed=seed)
        self.lyric_clf_ = build_lyric_mlp_classifier(lyric_dim, seed=seed)
        self.meta_ = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=seed,
        )
        self.classes_ = np.arange(config.NUM_CLASSES)

    def fit(self, X_audio, X_lyric, y, X_audio_val=None, X_lyric_val=None, y_val=None):
        print("  [LateFusion-Learnable] Training audio model …")
        self.audio_clf_.fit(X_audio, y, X_audio_val, y_val)
        print("  [LateFusion-Learnable] Training lyric model …")
        self.lyric_clf_.fit(X_lyric, y, X_lyric_val, y_val)

        if X_audio_val is None or X_lyric_val is None or y_val is None:
            raise ValueError("Learnable late fusion requires validation split.")

        pa = self.audio_clf_.predict_proba(X_audio_val)
        pl = self.lyric_clf_.predict_proba(X_lyric_val)
        self.meta_.fit(np.concatenate([pa, pl], axis=1), y_val)
        return self

    def predict_proba(self, X_audio, X_lyric):
        pa = self.audio_clf_.predict_proba(X_audio)
        pl = self.lyric_clf_.predict_proba(X_lyric)
        return self.meta_.predict_proba(np.concatenate([pa, pl], axis=1))

    def predict(self, X_audio, X_lyric):
        return np.argmax(self.predict_proba(X_audio, X_lyric), axis=1)
