"""Step 4b: the "actions" expert (MViT-v2 video model), tested in every brain window.

Feature sets: 4 depths x 2 ways of using the clips
    B3 / B9 / B14 / B16   = block 3, 9, 14 or 16 of the model (shallow -> deep)
    _full                 = one clip covering the whole 3 s
    _halves               = first-half clip and second-half clip side by side,
                            so the model can tell what happened early vs late

    python 07_motion_expert.py
"""
from __future__ import annotations

import numpy as np

from tsx.experts import run_expert

z = np.load("cache/motion.npz")
clips = [str(c) for c in z["clips"]]

feature_sets = {}
for b in z["blocks"]:
    name = f"B{int(b) + 1}"
    F = z[f"block{int(b) + 1}"].astype(np.float32)          # (1102, 3 clips, d)
    feature_sets[f"{name}_full"] = F[:, clips.index("full")]
    feature_sets[f"{name}_halves"] = np.concatenate(
        [F[:, clips.index("first")], F[:, clips.index("second")]], axis=1)

run_expert("motion", feature_sets, z["ids"])
