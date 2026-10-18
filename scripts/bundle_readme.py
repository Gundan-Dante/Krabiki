"""Write BUNDLE_README.md for an exported inference bundle, from its bundle_meta.json and one weights file.

Example:
    python scripts/bundle_readme.py --bundle bundle/krabiki_final_v4_bundle --title "final v4" \\
        --dev "sub-01..04 mean 33.1 / 67.9" --expected-dir submissions/final_v4_csls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEADS = {"decw": "ridge decoder, noise-whitened input", "dec": "ridge decoder, raw input",
         "pdecw": "subject-pooled ridge decoder, noise-whitened input (other subjects at weight λ)",
         "con": "contrastive two-tower head (InfoNCE), mean over seeds"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--dev", default="", help="dev result line, e.g. 'mean 33.1 / 67.9 / 2.9'")
    p.add_argument("--expected-dir", default="", help="committed submission dir the bundle reproduces")
    p.add_argument("--verified", default="", help="verification summary line")
    a = p.parse_args()
    meta = json.loads((a.bundle / "bundle_meta.json").read_text())
    w = np.load(a.bundle / "weights" / "sub-01.npz")
    members = meta["members"]
    sw = meta["weights"]
    same = all(sw[k] == sw["1"] for k in sw)
    lines = [f"# Krabiki inference bundle: {a.title}", "",
             f"Fitted parameters of the ensemble `{meta.get('source', '')}`, packaged so the ranked candidate lists can be "
             "produced **on a CPU-only machine (e.g. a MacBook) with numpy and scipy only**. No GPU and no torch / CLIP / "
             "DINOv2 / V-JEPA / CLAP; the 102 test videos' features are included.", ""]
    if a.dev:
        lines += [f"Dev score (sub-01..04): **{a.dev}** (chance 1.0% / 4.9%).", ""]
    lines += ["## Run it", "", "Requirements: Python 3.11, `numpy==2.4.6`, `scipy==1.17.1` (repo `requirements.txt`; nearby versions "
              "should work) and the challenge data folder with `tst/sub-XX_tst.npy`.", "", "```bash",
              f"unzip {a.bundle.name}.zip",
              f"python {a.bundle.name}/infer_bundle.py --bundle {a.bundle.name} --root /path/to/eeg-mind-reading-challenge "
              "--subjects 5 6 --out my_submission", "```", "",
              "Output: `my_submission/sub-0X_ranked_candidate_list.npy`, int64 (102, 102): row i = the i-th recording of "
              "`tst/sub-0X_tst.npy` in its original order, values = all video IDs 1001..1102 from most to least likely. "
              "`--subjects 1 2 3 4 5 6` runs every subject and prints top-1 / top-5 for sub-01..04 when `tst_stim/` is present.", ""]
    if a.expected_dir or a.verified:
        lines += [f"**Verified:** {a.verified or 'identical to ' + a.expected_dir}", ""]
    lines += ["## What the script does", "", "```",
              "tst/sub-XX_tst.npy (102 x 17 x 1500 @ 500 Hz)",
              f"  -> crop 0-{meta['max_seconds']} s -> scipy.signal.decimate(x, {meta['decimate_factor']}, ftype='fir', zero_phase=True)",
              "  -> center-test: mean over the 102 recordings -> train_eeg_mean" if meta["center_test"] else "  -> (no centering)",
              f"  -> per member: window samples {meta['window'][0]}:{meta['window'][1]} -> per-timepoint whitening (whiteners) "
              "-> flatten channel-major -> (x - x_mean) / x_std",
              "       ridge members (decw / dec / pdecw): pred = X @ B + y_mean; cosine vs (candidates - f_mean) / f_std",
              "       contrastive (con): per seed cosine(X @ W_e, cand @ W_v), row-z, mean over seeds",
              (f"  -> CSLS k={meta['csls_k']} per member -> row-z -> weighted sum" if meta["csls_k"] and meta["csls_stage"] == "member"
               else f"  -> row-z -> weighted sum -> CSLS k={meta['csls_k']} on the blend" if meta["csls_k"] else "  -> row-z -> weighted sum"),
              "  -> rank candidates per row (descending, stable) -> IDs 1001..1102", "```", "",
              "All fitted quantities come from the subject's training recordings (pooled members: plus the other subjects' "
              "training recordings at weight λ); test data enter only at scoring; centering and CSLS use only the unlabeled test set.", "",
              "## Members and weights", "", "| # | member | head | features (frames) | weight" + ("" if same else " (sub-01..06)") + " |",
              "|---|---|---|---|---|"]
    for i, m in enumerate(members):
        wt = f"{sw['1'][i]:.2f}" if same else " / ".join(f"{sw[str(s)][i]:.2f}" for s in range(1, 7))
        lines.append(f"| {i} | `{m['spec']}` | {HEADS.get(m['head'], m['head'])} | {m['features']} ({m['frames']}) | {wt} |")
    lines += ["", "Per-subject alphas (and λ for pooled members) are in `bundle_meta.json[\"checks\"]` and in each weights file "
              "(`<member>/dec_alpha`, `<member>/enc_alpha`, `<member>/lam`).", "",
              "## Files", "", "```", f"{a.bundle.name}/", "  BUNDLE_README.md, infer_bundle.py, bundle_meta.json",
              "  weights/sub-01.npz .. sub-06.npz   per-subject parameters (schema below)",
              "  candidates/<member>.npy            (102, D) float32 test-video features, row j = video 1001 + j",
              "  ARCHITECTURE.md, IMPROVEMENTS.md, EEGNET.md, schematics/   documentation and figures", "```", "",
              "**`weights/sub-01.npz`** (all subjects have the same keys):", "", "| array | shape | dtype |", "|---|---|---|"]
    for k in sorted(w.files):
        lines.append(f"| `{k}` | {w[k].shape} | {w[k].dtype} |")
    lines += ["", "Meaning: `train_eeg_mean` = mean preprocessed training EEG (center-test); `whiteners` = per-timepoint "
              "Σ_t^-1/2 (Ledoit-Wolf on encoding out-of-fold training residuals); `x_mean` / `x_std` = standardization of the "
              "whitened window; `f_mean` / `f_std` = training-video feature statistics (candidates); `B`, `y_mean` = ridge map "
              "and intercept at the chosen alpha (pooled: from the weighted pooled ridge at the chosen λ); `seed*/W_e`, `W_v`, "
              "`tau` = contrastive projections.", "",
              "## Provenance", "", "CLIP ViT-B-32 `laion2b_s34b_b79k` (open_clip); DINOv2 ViT-B/14 timm `vit_base_patch14_dinov2.lvd142m`; "
              "V-JEPA 2 `facebook/vjepa2-vitl-fpc64-256` rev `b3c1679b…`; AlexNet torchvision `IMAGENET1K_V1`; LAION-CLAP "
              "`laion/clap-htsat-unfused` rev `8fa0f1c6…`; hand-crafted / low-level AV descriptors computed by our code. "
              f"Parameters exported by `scripts/export_bundle.py` from the producing commit; checks in `bundle_meta.json`."]
    (a.bundle / "BUNDLE_README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {a.bundle / 'BUNDLE_README.md'}")


if __name__ == "__main__":
    main()
