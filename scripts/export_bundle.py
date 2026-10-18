"""Export the fitted parameters of the headline ensemble (final_csls = fin5_ctr_m5) into an inference bundle.

Runs the AUDITED pipeline code (``--code-dir``, a checkout of track-matching 7297a59) to refit every member per subject
with the alphas stored in the ensemble's similarity cache, then writes numpy-only parameters that
``scripts/infer_bundle.py`` turns back into the same similarity matrices. Every member is checked twice: against the
live pipeline object and against the cached test similarity the headline used.

Example:
    python scripts/export_bundle.py --code-dir /home/kparpie/Krabiki-audit \\
        --cache /home/kparpie/Krabiki-matching/results/similarity/_cache --out bundle/krabiki_final_csls_bundle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# member name, head, feature cache, frames, global weight (headline final_csls, LOFO-chosen)
MEMBERS = [("decw_clip_b32", "decw", "clip_b32", 8, 0.0),
           ("decw_dinov2_b_patch_all", "decw", "dinov2_b_patch_all", 8, 0.4),
           ("decw_vjepa2_l_all", "decw", "vjepa2_l_all", 16, 0.3),
           ("con_clip_b32_two_tower_d256", "con", "clip_b32", 8, 0.2),
           ("decw_clip_b32_mid8_all", "decw", "clip_b32_mid8_all", 8, 0.1)]
WINDOW, MAX_SECONDS, CSLS_K, CON_SEEDS = (5, 60), 1.5, 5, (0, 1, 2)


def import_audited(code_dir: Path):
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(here)]
    sys.path[:0] = [str(code_dir / "src"), str(code_dir / "scripts")]
    import ensemble  # noqa: E402

    assert Path(ensemble.__file__).resolve().parent == (code_dir / "scripts").resolve()
    return ensemble


def cache_file(cache: Path, s: int, head: str, feats: str, frames: int, kind: str) -> Path:
    if head == "con":
        return cache / f"sub-{s:02d}_con-{feats}-two_tower_d256_f{frames}_t1.5_w5-60_seeds0-2_ctr_{kind}.npz"
    return cache / f"sub-{s:02d}_decw-{feats}_f{frames}_t1.5_w5-60_ctr_{kind}.npz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code-dir", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("eeg-mind-reading-challenge"))
    p.add_argument("--feature-dir", type=Path, default=Path("features"))
    a = p.parse_args()
    root, feature_dir = a.root.resolve(), a.feature_dir.resolve()
    ens = import_audited(a.code_dir.resolve())
    import torch

    from eegmr import gpu
    from eegmr.contrastive import prepare, train_head
    from eegmr.data import TEST_IDS, load_test, load_train
    from eegmr.decoding import DecodingVariant, decoder_input, noise_whiteners
    from eegmr.features import load_features

    window = slice(*WINDOW)
    (a.out / "weights").mkdir(parents=True, exist_ok=True)
    (a.out / "candidates").mkdir(parents=True, exist_ok=True)
    for name, head, feats, frames, _ in MEMBERS:
        np.save(a.out / "candidates" / f"{name}.npy", load_features(feats, frames, TEST_IDS, feature_dir).astype(np.float32))
    cfg = ens.contrastive_configs()["two_tower_d256"]
    report = {}
    for s in range(1, 7):
        X, ids = load_train(s, root)
        eeg = ens.preprocess(X, MAX_SECONDS)
        measured = ens.center_like(ens.preprocess(load_test(s, root), MAX_SECONDS), eeg)
        out = {"train_eeg_mean": eeg.mean(axis=0).astype(np.float32)}
        for name, head, feats, frames, _ in MEMBERS:
            with np.load(cache_file(a.cache, s, head, feats, frames, "cv")) as z:
                dec_alpha, enc_alpha = (float(x) for x in z["alphas"])
            with np.load(cache_file(a.cache, s, head, feats, frames, "test")) as z:
                cached = z["sim"]
            F_trn = load_features(feats, frames, ids, feature_dir)
            F_tst = load_features(feats, frames, TEST_IDS, feature_dir)
            f_mean, f_std = F_trn.astype(np.float64).mean(0), F_trn.astype(np.float64).std(0)
            f_std = np.where(f_std == 0, 1.0, f_std)
            out[f"{name}/dec_alpha"], out[f"{name}/enc_alpha"] = np.float64(dec_alpha), np.float64(enc_alpha)
            if head == "decw":
                dec = DecodingVariant(dec_alpha, whiten_input=True, enc_alpha=enc_alpha, window=window).fit(F_trn, eeg)
                live = dec.similarity(measured, F_tst)
                b = dec.basis_
                Y = gpu.standardize(dec.features_trn_, dec.features_trn_)
                y_mean = Y.mean(dim=0)
                B = (b.V * (b.S / (b.S**2 + dec_alpha))) @ (b.U.T @ (Y - y_mean))  # (C*T_w, D), float64
                out |= {f"{name}/whiteners": dec.whiteners_.cpu().numpy().astype(np.float32),
                        f"{name}/x_mean": b.mean.cpu().numpy().astype(np.float32),
                        f"{name}/x_std": b.std.cpu().numpy().astype(np.float32),
                        f"{name}/B": B.cpu().numpy().astype(np.float32),
                        f"{name}/y_mean": y_mean.cpu().numpy().astype(np.float32)}
            else:
                dev = gpu.default_device()
                W = noise_whiteners(F_trn, eeg, enc_alpha, window, device=dev)
                X_trn, X_tst, V_trn, V_tst = prepare(eeg, measured, F_trn, F_tst, W, window, dev)
                Xr = decoder_input(eeg, window, W, dev)
                x_std = Xr.std(dim=0, unbiased=False)
                out |= {f"{name}/whiteners": W.cpu().numpy().astype(np.float32),
                        f"{name}/x_mean": Xr.mean(dim=0).cpu().numpy().astype(np.float32),
                        f"{name}/x_std": torch.where(x_std == 0, torch.ones_like(x_std), x_std).cpu().numpy().astype(np.float32)}
                z = 0
                for seed in CON_SEEDS:
                    h = train_head(X_trn, V_trn, cfg, dec_alpha, seed)
                    z = z + ens.row_z(h.similarity(X_tst, V_tst).double().cpu().numpy())
                    out |= {f"{name}/seed{seed}/W_e": h.W_e.cpu().numpy().astype(np.float32),
                            f"{name}/seed{seed}/W_v": h.W_v.cpu().numpy().astype(np.float32),
                            f"{name}/seed{seed}/tau": np.float32(h.log_tau.exp().item())}
                live = z / len(CON_SEEDS)
            out[f"{name}/f_mean"], out[f"{name}/f_std"] = f_mean.astype(np.float32), f_std.astype(np.float32)
            report[f"sub-{s:02d}/{name}"] = {"live_vs_cached_max_abs": float(np.abs(live - cached).max()),
                                             "dec_alpha": dec_alpha, "enc_alpha": enc_alpha}
        np.savez(a.out / "weights" / f"sub-{s:02d}.npz", **out)
        print(f"sub-{s:02d} exported", flush=True)
    meta = {"members": [{"name": n, "head": h, "features": f, "frames": fr, "weight": w} for n, h, f, fr, w in MEMBERS],
            "window": list(WINDOW), "max_seconds": MAX_SECONDS, "sfreq": 500, "decimate_factor": 5, "csls_k": CSLS_K,
            "con_seeds": list(CON_SEEDS), "center_test": True, "candidate_ids": [1001, 1102],
            "source": "track-matching 7297a59, experiment fin5_ctr_m5 (submissions/final_csls)", "checks": report}
    (a.out / "bundle_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
