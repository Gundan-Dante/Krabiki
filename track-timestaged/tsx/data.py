"""Loading the challenge data (EEG recordings, labels, video paths)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import decimate

SFREQ = 500                      # Hz, raw sampling rate
N_CHANNELS = 17
N_SAMPLES = 1500                 # 3 s at 500 Hz
TEST_IDS = np.arange(1001, 1103) # the 102 candidate videos, in column order
DEV_SUBJECTS = (1, 2, 3, 4)      # test labels available (validation only)
HELDOUT_SUBJECTS = (5, 6)        # test labels withheld (scored by organizers)
SUBJECTS = DEV_SUBJECTS + HELDOUT_SUBJECTS

def data_root(root: str | Path | None = None) -> Path:
    """Return the data folder.

    Order of preference: the `root` argument, the EEG_DATA_ROOT environment
    variable, then `eeg-mind-reading-challenge/` next to this track's folder.
    """
    if root is None:
        root = os.environ.get("EEG_DATA_ROOT") or (
            Path(__file__).resolve().parents[2] / "eeg-mind-reading-challenge"
        )
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data folder not found: {root}. Set EEG_DATA_ROOT to the "
            "eeg-mind-reading-challenge folder."
        )
    return root

def load_train(subject: int, root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (video_ids, eeg) for a subject's training data.

    video_ids: (n,) ints, sorted ascending.
    eeg:       (n, 17, 1500) float32, row k belongs to video_ids[k].
    """
    path = data_root(root) / "trn" / f"sub-{subject:02d}_trn.npz"
    with np.load(path) as z:
        ids = np.array(sorted(z.files, key=int), dtype=int)
        eeg = np.stack([z[str(i)] for i in ids]).astype(np.float32)
    _check_eeg(eeg, path)
    return ids, eeg

def load_test(subject: int, root: str | Path | None = None) -> np.ndarray:
    """Return test EEG (102, 17, 1500) in the ORIGINAL file order. Never reorder it."""
    path = data_root(root) / "tst" / f"sub-{subject:02d}_tst.npy"
    eeg = np.load(path).astype(np.float32)
    _check_eeg(eeg, path)
    if len(eeg) != len(TEST_IDS):
        raise ValueError(f"{path}: expected {len(TEST_IDS)} recordings, got {len(eeg)}")
    return eeg

def load_test_labels(subject: int, root: str | Path | None = None) -> np.ndarray | None:
    """Return true video IDs of the test recordings (same order as load_test), or
    None for held-out subjects. Use for validation only, never for training."""
    path = data_root(root) / "tst_stim" / f"sub-{subject:02d}_tst_stim.npy"
    if not path.exists():
        return None
    return np.load(path).astype(int)

def video_path(video_id: int, root: str | Path | None = None) -> Path:
    """Path of the .mp4 for a video ID (1-1000 training, 1001-1102 test)."""
    folder = "stimuli_train" if video_id <= 1000 else "stimuli_test"
    return data_root(root) / folder / f"{video_id:04d}.mp4"

def downsample(eeg: np.ndarray, factor: int = 5) -> np.ndarray:
    """Anti-aliased downsampling along time (default 500 Hz -> 100 Hz)."""
    return decimate(eeg, factor, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)

def _check_eeg(eeg: np.ndarray, path: Path) -> None:
    if eeg.ndim != 3 or eeg.shape[1:] != (N_CHANNELS, N_SAMPLES):
        raise ValueError(f"{path}: expected (n, {N_CHANNELS}, {N_SAMPLES}), got {eeg.shape}")
    if not np.isfinite(eeg).all():
        raise ValueError(f"{path}: contains NaN or inf")
