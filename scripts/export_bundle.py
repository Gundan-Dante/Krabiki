"""Export the fitted parameters of a CV-weighted ensemble (``scripts/ensemble.py``) into a numpy-only inference bundle.

Runs the AUDITED pipeline code (``--code-dir``, a checkout of the commit that produced the submission) to refit every
member per subject with the alphas stored in the ensemble's similarity cache (or re-chosen by the same training-only CV
if a cache entry is missing). Writes parameters that ``scripts/infer_bundle.py`` turns back into the same similarity
matrices. Every member is checked against the live pipeline object, and against the cached test similarity when present.

Members use ``ensemble.py``'s syntax: ``decw:FEATURES[:FRAMES]`` / ``dec:...`` (ridge decoders, any feature cache, e.g.
CLIP / DINOv2 / V-JEPA / AlexNet maps / hand-crafted descriptors / CLAP / low-level AV) and
``con:FEATURES:CONFIG[:FRAMES]`` (contrastive heads, averaged over ``--con-seeds``). Weights: ``--weights`` (one global
vector, integers summing to 10 or fractions) or ``--subject-weights`` (JSON ``{"1": [...], ...}``) for per-subject weights.

Example (the headline final_csls):
    python scripts/export_bundle.py --code-dir /home/kparpie/Krabiki-audit \\
        --cache /home/kparpie/Krabiki-matching/results/similarity/_cache --out bundle/krabiki_final_csls_bundle \\
        --member decw:clip_b32 --member decw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 \\
        --member con:clip_b32:two_tower_d256 --member decw:clip_b32_mid8_all:8 --weights 0 4 3 2 1 \\
        --center-test --csls 5 --csls-stage member --source "track-matching 7297a59, fin5_ctr_m5 (final_csls)"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def import_audited(code_dir: Path):
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(here)]
    sys.path[:0] = [str(code_dir / "src"), str(code_dir / "scripts")]
    import ensemble  # noqa: E402

    assert Path(ensemble.__file__).resolve().parent == (code_dir / "scripts").resolve()
    return ensemble


def bundle_name(m) -> str:
    return f"{m.name.replace(':', '-')}_f{m.frames}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code-dir", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=None, help="the ensemble's --sim-cache directory (alphas + checks)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--member", action="append", required=True)
    p.add_argument("--weights", type=float, nargs="+", default=None)
    p.add_argument("--subject-weights", type=Path, default=None, help='JSON {"1": [w, ...], ...}')
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=1.5)
    p.add_argument("--center-test", action="store_true")
    p.add_argument("--csls", type=int, default=0)
    p.add_argument("--csls-stage", choices=["member", "blend"], default="member")
    p.add_argument("--con-seeds", type=int, default=3)
    p.add_argument("--con-seed-start", type=int, default=0)
    p.add_argument("--source", default="", help="provenance note (commit, experiment)")
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

    members = [ens.parse_member(spec, a.frames) for spec in a.member]
    for m in members:
        assert m.head in ("decw", "dec", "con"), f"export supports decw / dec / con members, not {m.head!r}"
    seeds = list(range(a.con_seed_start, a.con_seed_start + a.con_seeds))
    if a.subject_weights:
        weights = {int(k): [float(x) for x in v] for k, v in json.loads(a.subject_weights.read_text()).items()}
    else:
        weights = {s: [float(x) for x in a.weights] for s in a.subjects}
    for s, w in weights.items():
        assert len(w) == len(members), f"sub-{s}: {len(w)} weights for {len(members)} members"
        total = sum(w)
        weights[s] = [x / total for x in w]  # integers summing to 10 or fractions: both normalized to sum 1
    window = slice(*a.window)
    cache_args = argparse.Namespace(sim_cache=a.cache, max_seconds=a.max_seconds, window=a.window,
                                    con_seeds=range(seeds[0], seeds[-1] + 1), center_test=a.center_test)
    (a.out / "weights").mkdir(parents=True, exist_ok=True)
    (a.out / "candidates").mkdir(parents=True, exist_ok=True)
    for m in members:
        np.save(a.out / "candidates" / f"{bundle_name(m)}.npy",
                load_features(m.features, m.frames, TEST_IDS, feature_dir).astype(np.float32))
    report = {}
    for s in a.subjects:
        X, ids = load_train(s, root)
        eeg = ens.preprocess(X, a.max_seconds)
        measured = ens.preprocess(load_test(s, root), a.max_seconds)
        if a.center_test:
            measured = ens.center_like(measured, eeg)
        out = {"train_eeg_mean": eeg.mean(axis=0).astype(np.float32)}
        for m in members:
            n = bundle_name(m)
            F_trn = load_features(m.features, m.frames, ids, feature_dir)
            F_tst = load_features(m.features, m.frames, TEST_IDS, feature_dir)
            cv_path = ens.cache_path(cache_args, s, m, "cv") if a.cache else None
            if cv_path is not None and cv_path.exists():
                with np.load(cv_path) as z:
                    dec_alpha, enc_alpha = (None if np.isnan(x) else float(x) for x in z["alphas"])
            else:
                dec_alpha, enc_alpha = ens.member_alphas(m, F_trn, eeg, window)
            test_path = ens.cache_path(cache_args, s, m, "test") if a.cache else None
            cached = np.load(test_path)["sim"] if test_path is not None and test_path.exists() else None
            f_mean, f_std = F_trn.astype(np.float64).mean(0), F_trn.astype(np.float64).std(0)
            out[f"{n}/f_mean"], out[f"{n}/f_std"] = f_mean.astype(np.float32), np.where(f_std == 0, 1.0, f_std).astype(np.float32)
            out[f"{n}/dec_alpha"] = np.float64(dec_alpha)
            out[f"{n}/enc_alpha"] = np.float64(np.nan if enc_alpha is None else enc_alpha)
            if m.head in ("decw", "dec"):
                dec = DecodingVariant(dec_alpha, whiten_input=m.head == "decw", enc_alpha=enc_alpha, window=window)
                dec.fit(F_trn, eeg)
                live = dec.similarity(measured, F_tst)
                b = dec.basis_
                Y = gpu.standardize(dec.features_trn_, dec.features_trn_)
                y_mean = Y.mean(dim=0)
                B = (b.V * (b.S / (b.S**2 + dec_alpha))) @ (b.U.T @ (Y - y_mean))
                W = dec.whiteners_
                out |= {f"{n}/x_mean": b.mean.cpu().numpy().astype(np.float32),
                        f"{n}/x_std": b.std.cpu().numpy().astype(np.float32),
                        f"{n}/B": B.cpu().numpy().astype(np.float32), f"{n}/y_mean": y_mean.cpu().numpy().astype(np.float32)}
            else:
                dev = gpu.default_device()
                cfg = ens.contrastive_configs()[m.config]
                W = noise_whiteners(F_trn, eeg, enc_alpha, window, device=dev)
                X_trn, X_tst, V_trn, V_tst = prepare(eeg, measured, F_trn, F_tst, W, window, dev)
                Xr = decoder_input(eeg, window, W, dev)
                x_std = Xr.std(dim=0, unbiased=False)
                out |= {f"{n}/x_mean": Xr.mean(dim=0).cpu().numpy().astype(np.float32),
                        f"{n}/x_std": torch.where(x_std == 0, torch.ones_like(x_std), x_std).cpu().numpy().astype(np.float32)}
                z = 0
                for seed in seeds:
                    h = train_head(X_trn, V_trn, cfg, dec_alpha, seed)
                    z = z + ens.row_z(h.similarity(X_tst, V_tst).double().cpu().numpy())
                    out |= {f"{n}/seed{seed}/W_e": h.W_e.cpu().numpy().astype(np.float32),
                            f"{n}/seed{seed}/W_v": h.W_v.cpu().numpy().astype(np.float32),
                            f"{n}/seed{seed}/tau": np.float32(h.log_tau.exp().item())}
                live = z / len(seeds)
            if W is not None:
                out[f"{n}/whiteners"] = W.cpu().numpy().astype(np.float32)
            report[f"sub-{s:02d}/{n}"] = {"dec_alpha": dec_alpha, "enc_alpha": enc_alpha,
                                          "live_vs_cached_max_abs": None if cached is None else float(np.abs(live - cached).max())}
        np.savez(a.out / "weights" / f"sub-{s:02d}.npz", **out)
        print(f"sub-{s:02d} exported", flush=True)
    meta = {"members": [{"name": bundle_name(m), "spec": spec, "head": m.head, "features": m.features, "frames": m.frames}
                        for m, spec in zip(members, a.member)],
            "weights": {str(s): w for s, w in weights.items()}, "window": a.window, "max_seconds": a.max_seconds,
            "sfreq": 500, "decimate_factor": 5, "csls_k": a.csls, "csls_stage": a.csls_stage, "con_seeds": seeds,
            "center_test": a.center_test, "candidate_ids": [1001, 1102], "source": a.source, "checks": report}
    (a.out / "bundle_meta.json").write_text(json.dumps(meta, indent=1))
    worst = max((v["live_vs_cached_max_abs"] or 0.0) for v in report.values())
    missing = sum(v["live_vs_cached_max_abs"] is None for v in report.values())
    print(f"export checks: max |live - cached| = {worst:.3g} over {len(report) - missing} member fits ({missing} without cache)")


if __name__ == "__main__":
    main()
