"""Writing and validating the official submission files."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import TEST_IDS
from .evaluate import rank_candidates

def validate_ranking(ranked: np.ndarray, candidate_ids: np.ndarray = TEST_IDS) -> None:
    """Raise if `ranked` doesn't match the required format:
    integer array (102, 102), every row a permutation of the 102 video IDs."""
    ranked = np.asarray(ranked)
    n = len(candidate_ids)
    if ranked.shape != (n, n):
        raise ValueError(f"expected shape ({n}, {n}), got {ranked.shape}")
    if not np.issubdtype(ranked.dtype, np.integer):
        raise ValueError(f"expected an integer array, got {ranked.dtype}")
    expected = np.sort(np.asarray(candidate_ids))
    bad = [i for i in range(n) if not np.array_equal(np.sort(ranked[i]), expected)]
    if bad:
        raise ValueError(f"rows {bad[:5]} are not permutations of the candidate IDs")

def submission_path(subject: int, out_dir: str | Path) -> Path:
    return Path(out_dir) / f"sub-{subject:02d}_ranked_candidate_list.npy"

def write_submission(similarity: np.ndarray, subject: int, out_dir: str | Path) -> Path:
    """Rank, validate, save, then reload and re-validate the file.

    `similarity` rows must be in the original order of tst/sub-XX_tst.npy.
    """
    ranked = rank_candidates(similarity).astype(np.int64)
    validate_ranking(ranked)
    path = submission_path(subject, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ranked)
    reloaded = np.load(path)
    validate_ranking(reloaded)
    if not np.array_equal(reloaded, ranked):
        raise RuntimeError(f"{path} did not round-trip correctly")
    return path
