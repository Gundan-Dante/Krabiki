"""Step 0: check the data loads and the submission pipeline is correct.

Run from the track-timestaged folder:
    python 00_check_setup.py
If the data isn't at ../eeg-mind-reading-challenge, set EEG_DATA_ROOT first.
"""
from __future__ import annotations

import numpy as np

from tsx.data import (
    SUBJECTS, TEST_IDS, data_root, load_test, load_test_labels, load_train, video_path,
)
from tsx.evaluate import score_ranking
from tsx.submission import write_submission

OUT_DIR = "outputs/sanity"

def main() -> None:
    root = data_root()
    print(f"Data folder: {root}")
    print()

    # 1. Every subject's EEG loads with the right shapes.
    print("subject  train_trials  test_shape      test_labels")
    for s in SUBJECTS:
        ids, _ = load_train(s)
        tst = load_test(s)
        labels = load_test_labels(s)
        status = "yes" if labels is not None else "no (held out)"
        print(f"sub-{s:02d}   {len(ids):>12}  {str(tst.shape):<14}  {status}")
        if labels is not None and sorted(labels) != list(TEST_IDS):
            raise ValueError(f"sub-{s:02d}: test labels are not exactly 1001..1102")

    # 2. All 1,102 videos exist.
    missing = [v for v in range(1, 1103) if not video_path(v).exists()]
    print()
    print(f"Videos found: {1102 - len(missing)}/1102")
    if missing:
        print(f"  missing (first 10): {missing[:10]}")

    # 3. Submission pipeline sanity check on sub-01 (has labels).
    #    a) "Perfect" scores built from the answer key must give top-1 = 1.0.
    #       This only proves row order and file format are right; it is NOT a model.
    #    b) Random scores must give roughly chance (top-1 ~0.01, top-5 ~0.05).
    labels = load_test_labels(1)
    perfect = (labels[:, None] == TEST_IDS[None, :]).astype(float)
    random = np.random.default_rng(0).standard_normal((len(labels), len(TEST_IDS)))

    for name, sim in [("perfect (format check)", perfect), ("random (chance check)", random)]:
        path = write_submission(sim, subject=1, out_dir=f"{OUT_DIR}/{name.split()[0]}")
        result = score_ranking(np.load(path), labels)
        print()
        print(f"{name}: {result}  ->  {path}")

    assert score_ranking(np.load(f"{OUT_DIR}/perfect/sub-01_ranked_candidate_list.npy"),
                         labels)["top1"] == 1.0
    print()
    print("All checks passed.")

if __name__ == "__main__":
    main()
