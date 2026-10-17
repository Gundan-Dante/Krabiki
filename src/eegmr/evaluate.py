"""Retrieval metrics shared by all tracks."""

from __future__ import annotations

import numpy as np

from eegmr.data import TEST_IDS


def true_ranks(similarity: np.ndarray, true_ids: np.ndarray, candidate_ids: np.ndarray = TEST_IDS) -> np.ndarray:
    """1-based rank of the true video in each row of ``similarity`` (rows: recordings, cols: candidates).

    Ties count against us: a candidate scoring equal to the true one is ranked ahead of it.
    """
    col = np.searchsorted(candidate_ids, true_ids)
    assert np.array_equal(candidate_ids[col], true_ids), "true_ids must be among candidate_ids"
    true_score = similarity[np.arange(len(true_ids)), col]
    return (similarity >= true_score[:, None]).sum(axis=1)  # counts the true video itself -> 1-based


def evaluate(similarity: np.ndarray, true_ids: np.ndarray, candidate_ids: np.ndarray = TEST_IDS) -> dict[str, float]:
    """Top-1 / top-5 accuracy and median rank of the true video."""
    ranks = true_ranks(similarity, true_ids, candidate_ids)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "median_rank": float(np.median(ranks)),
    }
