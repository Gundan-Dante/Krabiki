"""Evaluate the audit controls (CPU only): determinism vs the committed headline, row permutation, label shuffle.

Example (paths as used in the audit of fin5_ctr_m5):
    python scripts/audit_compare.py --audit-dir /home/kparpie/Krabiki-audit --ref-cache results/similarity/_cache \\
        --ref-submissions /home/kparpie/Krabiki-matching/submissions --out results/audit/controls.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eegmr.data import load_test_labels

MEMBERS = ["decw-clip_b32_f8", "decw-dinov2_b_patch_all_f8", "decw-vjepa2_l_all_f16",
           "con-clip_b32-two_tower_d256_f8", "decw-clip_b32_mid8_all_f8"]


def true_ranks(ranked: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.argmax(ranked == y[:, None], axis=1) + 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--ref-cache", type=Path, required=True)
    p.add_argument("--ref-submissions", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 5])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ad, rep = a.audit_dir, {}

    # 3c / 3d: determinism of the recomputed member test similarities and of the final ranked lists
    for s in a.subjects:
        d = {}
        for m in MEMBERS:
            seeds = "_seeds0-2" if m.startswith("con") else ""
            name = f"sub-{s:02d}_{m}_t1.5_w5-60{seeds}_ctr_test.npz"
            new_f, ref_f = ad / "audit_out/cache_rerun" / name, a.ref_cache / name
            if not (new_f.exists() and ref_f.exists()):
                d[m] = "missing"
                continue
            new, ref = np.load(new_f)["sim"], np.load(ref_f)["sim"]
            d[m] = {"max_abs_diff": float(np.abs(new - ref).max()),
                    "top1_agree": float(np.mean(new.argmax(1) == ref.argmax(1))),
                    "bit_identical": bool(np.array_equal(new, ref))}
        new_sub = ad / f"submissions/audit_rerun_cv/sub-{s:02d}_ranked_candidate_list.npy"
        ref_sub = a.ref_submissions / ("final_csls" if s >= 5 else "fin5_ctr_m5_cv") / f"sub-{s:02d}_ranked_candidate_list.npy"
        if new_sub.exists() and ref_sub.exists():
            n, r = np.load(new_sub), np.load(ref_sub)
            d["final_ranked_lists"] = {"identical": bool(np.array_equal(n, r)), "rows_identical": int((n == r).all(1).sum()),
                                       "top1_agree": float(np.mean(n[:, 0] == r[:, 0])), "ref": str(ref_sub)}
            y = load_test_labels(s)
            if y is not None:
                d["final_ranked_lists"]["dev_top1_new_vs_ref"] = [float(np.mean(true_ranks(n, y) <= 1)),
                                                                 float(np.mean(true_ranks(r, y) <= 1))]
        rep[f"determinism_sub-{s:02d}"] = d

    # 3b: row permutation
    perm_f = ad / "audit_out/perms.json"
    if perm_f.exists():
        perms = {int(k): np.array(v) for k, v in json.loads(perm_f.read_text()).items()}
        for s, perm in perms.items():
            if not (ad / f"submissions/audit_perm_cv/sub-{s:02d}_ranked_candidate_list.npy").exists():
                rep[f"permutation_sub-{s:02d}"] = "not finished"
                continue
            base = np.load(ad / f"submissions/audit_rerun_cv/sub-{s:02d}_ranked_candidate_list.npy")
            permuted = np.load(ad / f"submissions/audit_perm_cv/sub-{s:02d}_ranked_candidate_list.npy")
            expected = base[perm]
            rep[f"permutation_sub-{s:02d}"] = {"output_equals_permuted_input": bool(np.array_equal(permuted, expected)),
                                              "rows_identical": int((permuted == expected).all(1).sum()),
                                              "top1_agree": float(np.mean(permuted[:, 0] == expected[:, 0]))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
