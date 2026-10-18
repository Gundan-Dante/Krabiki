"""Artifact-aware training (S3): unsupervised outlier scores for EEG *training* trials.

Per trial and channel, on the raw 500 Hz EEG of the crop the decoders use: log variance, log peak-to-peak amplitude,
kurtosis, log high-frequency (> 30 Hz) power and log line length (17 x 5 = 85 values). Per subject, pyod HBOS and a
one-class SVM are fitted on the standardized features (no labels, no test data). Each detector's score becomes a rank in
[0, 1] and the two ranks are averaged; the top ``frac`` of trials are flagged.

Flagged trials are then dropped from, or down-weighted in, the decoder's ridge fit (:class:`WeightedDecoding`).
Test trials are never flagged: every test recording still gets a ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import kurtosis, rankdata

from eegmr import gpu
from eegmr.data import SFREQ
from eegmr.decoding import DecodingVariant, decoder_input

HF_CUTOFF = 30.0  # Hz


def trial_features(raw: np.ndarray, sfreq: int = SFREQ, hf_cutoff: float = HF_CUTOFF) -> np.ndarray:
    """``(n, 17, T)`` raw EEG -> ``(n, 17 * 5)`` artifact features (see module docstring)."""
    x = np.asarray(raw, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / sfreq)
    eps = 1e-12
    feats = np.stack([np.log(x.var(axis=-1) + eps),
                      np.log(np.ptp(x, axis=-1) + eps),
                      kurtosis(x, axis=-1),
                      np.log(power[..., freqs > hf_cutoff].mean(axis=-1) + eps),
                      np.log(np.abs(np.diff(x, axis=-1)).sum(axis=-1) + eps)], axis=-1)
    return feats.reshape(len(x), -1)


def outlier_scores(features: np.ndarray, seed: int = 0) -> np.ndarray:
    """Average of the rank-normalized HBOS and one-class-SVM outlier scores, in [0, 1]; higher = more anomalous."""
    from pyod.models.hbos import HBOS
    from pyod.models.ocsvm import OCSVM

    z = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-12)
    scores = [HBOS().fit(z).decision_scores_, OCSVM().fit(z).decision_scores_]
    return np.mean([rankdata(s) / len(s) for s in scores], axis=0)


def flag_outliers(scores: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask of the ``frac`` highest-scoring trials (ties broken by order)."""
    n_flag = int(round(frac * len(scores)))
    flags = np.zeros(len(scores), dtype=bool)
    if n_flag:
        flags[np.argsort(-scores, kind="stable")[:n_flag]] = True
    return flags


@dataclass
class WeightedDecoding(DecodingVariant):
    """:class:`DecodingVariant` whose ridge fit uses per-trial weights (exact weighted ridge).

    Minimizes ``sum_i w_i ||y_i - b - x_i B||^2 + alpha ||B||^2`` with ``x`` standardized by weighted statistics. With all
    weights 1 it equals the unweighted decoder up to float rounding (tested). The noise whiteners stay unweighted.
    """

    sample_weight: np.ndarray | None = None

    def fit(self, features: np.ndarray, eeg: np.ndarray) -> "WeightedDecoding":
        super().fit(features, eeg)
        if self.sample_weight is None:
            return self
        dev = self.basis_.U.device
        X = decoder_input(eeg, self.window, self.whiteners_, dev)
        # raw weights (inliers 1): no rescaling, so the chosen ridge alpha keeps its meaning and weight 0 == dropping
        w = gpu.as_tensor(self.sample_weight, dev, torch.float64)
        self.w_ = w
        mean = (w[:, None] * X).sum(0) / w.sum()
        std = torch.sqrt((w[:, None] * (X - mean) ** 2).sum(0) / w.sum())
        std = torch.where(std == 0, torch.ones_like(std), std)
        U, S, Vh = torch.linalg.svd(w.sqrt()[:, None] * (X - mean) / std, full_matrices=False)
        self.basis_ = gpu.RidgeBasis(mean, std, U, S, Vh.T, False)
        return self

    def similarity(self, eeg: np.ndarray, candidate_features: np.ndarray) -> np.ndarray:
        if self.sample_weight is None:
            return super().similarity(eeg, candidate_features)
        dev = self.basis_.U.device
        X = decoder_input(eeg, self.window, self.whiteners_, dev)
        w = self.w_
        F = self.features_trn_
        f_mean = (w[:, None] * F).sum(0) / w.sum()
        f_std = torch.sqrt((w[:, None] * (F - f_mean) ** 2).sum(0) / w.sum())
        f_std = torch.where(f_std == 0, torch.ones_like(f_std), f_std)
        Y = (F - f_mean) / f_std  # targets standardized with the same weights, so weight 0 == dropping the trial
        y_mean = (w[:, None] * Y).sum(0) / w.sum()
        UtY = self.basis_.U.T @ (w.sqrt()[:, None] * (Y - y_mean))
        s = self.basis_.S
        pred = (self.basis_.project(X) * (s / (s ** 2 + self.dec_alpha))) @ UtY + y_mean
        cand = (gpu.as_tensor(candidate_features, dev, torch.float64) - f_mean) / f_std
        return gpu.cosine_similarity(pred.float(), cand.float()).cpu().numpy().astype(np.float64)
