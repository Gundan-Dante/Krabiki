"""Turning similarity scores into rankings, and scoring rankings."""
from __future__ import annotations

import numpy as np

from .data import TEST_IDS

def rank_candidates(similarity: np.ndarray, candidate_ids: np.ndarray = TEST_IDS) -> np.ndarray:
    """Convert a similarity matrix into ranked video IDs.

    similarity[i, j] = how well test recording i (original order) matches
    candidate_ids[j]. Higher is better. Returns (n, n_candidates) video IDs,
    best first. Row order is kept exactly as given.
    """
    sim = np.asarray(similarity, dtype=float)
    candidate_ids = np.asarray(candidate_ids)
    if sim.ndim != 2 or sim.shape[1] != len(candidate_ids):
        raise ValueError(f"similarity must be (n, {len(candidate_ids)}), got {sim.shape}")
    if not np.isfinite(sim).all():
        raise ValueError("similarity contains NaN or inf")
    order = np.argsort(-sim, axis=1, kind="stable")
    return candidate_ids[order]

def score_ranking(ranked: np.ndarray, true_ids: np.ndarray) -> dict[str, float]:
    """Top-1, top-5 and median rank of the true video (rank 1 = best)."""
    ranked = np.asarray(ranked)
    true_ids = np.asarray(true_ids)
    hits = ranked == true_ids[:, None]
    if not (hits.sum(axis=1) == 1).all():
        raise ValueError("every row must contain its true video ID exactly once")
    rank = hits.argmax(axis=1) + 1
    return {
        "top1": float((rank == 1).mean()),
        "top5": float((rank <= 5).mean()),
        "median_rank": float(np.median(rank)),
    }

def score_similarity(similarity: np.ndarray, true_ids: np.ndarray) -> dict[str, float]:
    """Shortcut: rank a similarity matrix and score it."""
    return score_ranking(rank_candidates(similarity), true_ids)
