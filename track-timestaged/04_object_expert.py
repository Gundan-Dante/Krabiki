"""Step 2b: the "objects" expert (DINOv2), tested in every brain window.

Feature sets: 3 layer depths x 2 ways of using the frames
    L4 / L8 / L12   = block 4, 8 or 12 of the model (shallow -> deep)
    _onset          = mean of the first 2 frames (0.19 s and 0.56 s)
    _all            = mean of all 8 frames

    python 04_object_expert.py
"""
from __future__ import annotations

import numpy as np

from tsx.experts import run_expert

z = np.load("cache/dino.npz")
F = z["features"].astype(np.float32)          # (1102, 8 frames, 3 layers, 768)
layers = [int(l) + 1 for l in z["layers"]]

feature_sets = {}
for li, layer in enumerate(layers):
    feature_sets[f"L{layer}_onset"] = F[:, :2, li].mean(axis=1)
    feature_sets[f"L{layer}_all"] = F[:, :, li].mean(axis=1)

run_expert("dino", feature_sets, z["ids"])
