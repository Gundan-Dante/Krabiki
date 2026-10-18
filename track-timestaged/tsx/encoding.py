"""Per-window encoding models: predict the EEG in a time window from video
features, then score candidates by how well the prediction matches the recording."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Brain-time windows in seconds after video onset (see RESEARCH_ALTERNATIVES.md).
WINDOWS: dict[str, tuple[float, float]] = {
    "early": (0.05, 0.25),
    "middle": (0.25, 0.80),
    "late": (0.80, 3.00),
}

def window_slice(t0: float, t1: float, sfreq: float) -> slice:
    """Sample indices for [t0, t1) seconds."""
    return slice(int(round(t0 * sfreq)), int(round(t1 * sfreq)))

def normalize_eeg(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each channel with the subject's training mean and std (no labels used)."""
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std

class WindowEncoder:
    """Ridge regression from video features (n, d) to EEG in one window (n, 17, w)."""

    def __init__(self, alphas: np.ndarray | None = None) -> None:
        self.alphas = np.logspace(0, 7, 15) if alphas is None else alphas

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "WindowEncoder":
        self.out_shape_ = Y.shape[1:]
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = RidgeCV(alphas=self.alphas).fit(
            self.scaler_.transform(X), Y.reshape(len(Y), -1)
        )
        return self

    @property
    def alpha(self) -> float:
        return float(self.model_.alpha_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.model_.predict(self.scaler_.transform(X))
        return pred.reshape(len(X), *self.out_shape_)

def correlation_similarity(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Pearson correlation between each measured recording (n, 17, w) and each
    predicted candidate response (m, 17, w). Returns (n, m)."""
    M = _zscore_rows(measured.reshape(len(measured), -1))
    P = _zscore_rows(predicted.reshape(len(predicted), -1))
    return M @ P.T / M.shape[1]

def combine(similarities: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted sum of similarity matrices, each z-scored per row first so no
    window dominates just because its numbers are bigger."""
    weights = [1.0] * len(similarities) if weights is None else weights
    return sum(w * _zscore_rows(s) for w, s in zip(weights, similarities))

def _zscore_rows(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-12)
