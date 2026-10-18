"""Linear encoding model: stimulus features -> predicted EEG, retrieval by correlation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_ALPHAS = np.logspace(0, 9, 37)  # quarter-decade steps
DEFAULT_WINDOW = slice(5, 60)  # 50-600 ms at 100 Hz


def correlation_similarity(measured: np.ndarray, predicted: np.ndarray, window: slice) -> np.ndarray:
    """Pearson correlation between every measured and every predicted EEG over ``window`` (channels x time flattened).

    Returns ``(n_measured, n_predicted)``.
    """

    def zscore(x: np.ndarray) -> np.ndarray:
        x = x[..., window].reshape(len(x), -1)
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return zscore(measured) @ zscore(predicted).T


def _feature_transform(pca_variance: float, whiten: bool = False) -> Pipeline:
    """Standardize -> PCA, optionally followed by a second standardization.

    ``whiten=True`` makes every PCA component unit-variance, so Z^T Z = n*I and all singular values are equal. Ridge
    then shrinks every direction by the same factor, alpha only rescales predictions, and correlation matching
    ignores it. Kept only to reproduce runs logged before 2026-10-18.
    """
    steps = [StandardScaler(), PCA(n_components=pca_variance, svd_solver="full")]
    return make_pipeline(*steps, *([StandardScaler()] if whiten else []))


def retrieval_cv_curve(
    features: np.ndarray,
    eeg: np.ndarray,
    alphas: np.ndarray,
    window: slice,
    pca_variance: float = 0.95,
    n_folds: int = 5,
    seed: int = 0,
    whiten: bool = False,
    erp_subtract: bool = False,
) -> np.ndarray:
    """Mean percentile rank (0 = always first, 0.5 = chance) of the true video on held-out training folds, per alpha.

    Each fold refits the feature transform on its own training part, then solves ridge for all alphas at once via
    one SVD. Held-out videos are retrieved among each other with the same correlation matching used at test time.
    With ``erp_subtract`` the fold's training-mean response is removed from the held-out and predicted EEG first.
    """
    scores = np.zeros(len(alphas))
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(features):
        transform = _feature_transform(pca_variance, whiten).fit(features[trn])
        Z_trn, Z_val = transform.transform(features[trn]), transform.transform(features[val])
        Y_trn = eeg[trn].reshape(len(trn), -1)
        y_mean = Y_trn.mean(axis=0)
        U, s, Vt = np.linalg.svd(Z_trn, full_matrices=False)  # Z_trn is column-centred: PCA output on its own fit data
        UtY, ZV = U.T @ (Y_trn - y_mean), Z_val @ Vt.T
        erp = y_mean.reshape(eeg.shape[1:])
        for k, alpha in enumerate(alphas):
            pred = (ZV * (s / (s**2 + alpha))) @ UtY + y_mean
            pred = pred.reshape(len(val), *eeg.shape[1:])
            if erp_subtract:
                sim = correlation_similarity(eeg[val] - erp, pred - erp, window)
            else:
                sim = correlation_similarity(eeg[val], pred, window)
            ranks = (sim >= np.diag(sim)[:, None]).sum(axis=1)  # true video is on the diagonal
            scores[k] += np.sum((ranks - 1) / (len(val) - 1))
    return scores / len(features)


@dataclass
class EncodingModel:
    """Standardize -> PCA (-> standardize if ``whiten``) -> ridge, predicting the full (channels x time) EEG.

    ``alpha_selection``:
        ``"gcv"``        RidgeCV's generalized cross-validation (minimizes squared error over all EEG targets).
        ``"retrieval"``  k-fold CV on the training videos, maximizing retrieval within ``window``.
    """

    pca_variance: float = 0.95
    alphas: np.ndarray = field(default_factory=lambda: DEFAULT_ALPHAS)
    alpha_selection: Literal["gcv", "retrieval"] = "gcv"
    window: slice = field(default_factory=lambda: DEFAULT_WINDOW)
    n_folds: int = 5
    seed: int = 0
    whiten: bool = False
    erp_subtract: bool = False  # alpha selection only: match on EEG minus the (fold-)training mean response
    pipeline: Pipeline | None = None
    eeg_shape: tuple[int, ...] = ()
    cv_curve: np.ndarray | None = None

    def fit(self, features: np.ndarray, eeg: np.ndarray) -> "EncodingModel":
        """``features``: ``(n, D)``; ``eeg``: ``(n, channels, time)``."""
        self.eeg_shape = eeg.shape[1:]
        if self.alpha_selection == "gcv":
            ridge = RidgeCV(alphas=self.alphas)
        else:
            self.cv_curve = retrieval_cv_curve(features, eeg, self.alphas, self.window, self.pca_variance,
                                               self.n_folds, self.seed, self.whiten, self.erp_subtract)
            ridge = Ridge(alpha=float(self.alphas[np.argmin(self.cv_curve)]))
        self.pipeline = Pipeline(_feature_transform(self.pca_variance, self.whiten).steps + [("ridge", ridge)])
        self.pipeline.fit(features, eeg.reshape(len(eeg), -1))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(features).reshape(len(features), *self.eeg_shape)

    @property
    def alpha(self) -> float:
        ridge = self.pipeline[-1]
        return float(ridge.alpha_ if isinstance(ridge, RidgeCV) else ridge.alpha)

    @property
    def n_components(self) -> int:
        return int(self.pipeline[1].n_components_)
