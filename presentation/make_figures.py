"""Regenerate every presentation figure from saved outputs.

Run from the repo root:
    PYTHONPATH=src python presentation/make_figures.py [fig1 fig3 ...]     (no argument = all)

Inputs: presentation/data/ (copies of saved similarity matrices and CV tables from the model tracks), the training
EEG, the cached features/ and the test videos. Nothing is re-trained except two cheap diagnostics (figures 3 and 4b).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from eegmr.data import DEV_SUBJECTS, TEST_IDS, decimate, load_test_labels, load_train, video_path
from eegmr.evaluate import true_ranks

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "figures"
N_CAND = len(TEST_IDS)
CHANCE_TOP1, CHANCE_TOP5 = 1 / N_CAND, 5 / N_CAND

# Okabe-Ito (colour-blind safe). Colour = kind of model, never rank.
C_ENC, C_DEC, C_ENS, C_DINO = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E4E4E4"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 15, "axes.titlesize": 19, "axes.titleweight": "bold", "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})

# name -> (folder under presentation/data/sim, colour)
MODELS = {
    "enc": "enc_clip", "enc_noise": "enc_clip_noise_euclid", "dec": "dec_clip", "decw": "decw_clip",
    "dino_patch": "decw_dino_patch", "dino_cls": "decw_dino_cls", "ens": "ens_clip_dino_patch",
}


# ---------------------------------------------------------------- shared helpers
def similarity(model: str, subject: int) -> np.ndarray:
    return np.load(DATA / "sim" / MODELS[model] / f"sub-{subject:02d}.npy")


def dev_ranks(model: str) -> np.ndarray:
    """True-video ranks pooled over the four dev subjects (408 trials)."""
    return np.concatenate([true_ranks(similarity(model, s), load_test_labels(s)) for s in DEV_SUBJECTS])


def boot_ci(ranks: np.ndarray, k: int, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of top-k accuracy over trials."""
    rng = np.random.default_rng(seed)
    hit = (ranks <= k).astype(float)
    means = hit[rng.integers(0, len(hit), (n, len(hit)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def alexnet_dev() -> dict[str, np.ndarray]:
    """M0 has no saved similarity matrix; its per-subject metrics are in results.csv."""
    df = pd.read_csv(DATA / "results.csv")
    df = df[(df.experiment == "enc_alexnet_f8") & df.subject.isin(DEV_SUBJECTS)]
    return {"top1": df.top1.to_numpy(), "top5": df.top5.to_numpy()}


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def chance_line(ax, y: float, label: str | None = None, horizontal: bool = True) -> None:
    (ax.axhline if horizontal else ax.axvline)(y, color=MUTED, ls="--", lw=1.6, zorder=1)
    if label:
        ax.text(ax.get_xlim()[1], y, f" {label}", color=MUTED, va="bottom", ha="right", fontsize=13)


# ---------------------------------------------------------------- figure 1: progress
def fig1_progress() -> None:
    steps = [  # label, model key (None = AlexNet from results.csv), colour
        ("AlexNet\nencoding\n(baseline)", None, C_ENC),
        ("CLIP\nencoding", "enc", C_ENC),
        ("+ noise-\naware\nmatching", "enc_noise", C_ENC),
        ("CLIP\ndecoding", "dec", C_DEC),
        ("+ noise-\nwhitened\ninput", "decw", C_DEC),
        ("+ DINOv2\nensemble", "ens", C_ENS),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6))
    for ax, (k, key, chance) in zip(axes, [(1, "top1", CHANCE_TOP1), (5, "top5", CHANCE_TOP5)]):
        for i, (_, model, color) in enumerate(steps):
            if model is None:
                val = alexnet_dev()[key].mean()
                lo = hi = np.nan  # only per-subject means available for M0
            else:
                r = dev_ranks(model)
                val = (r <= k).mean()
                lo, hi = boot_ci(r, k)
            ax.bar(i, val * 100, color=color, width=0.72, zorder=3)
            if model is not None:
                ax.errorbar(i, val * 100, yerr=[[val * 100 - lo * 100], [hi * 100 - val * 100]], color=INK, capsize=4,
                            lw=1.6, zorder=4)
            ax.text(i, (hi if model else val) * 100 + 1, f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=14,
                    fontweight="bold")
        ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=11)
        ax.set_ylabel(f"Correct video in top {k} (%)")
        ax.set_title("Top-1: exact match" if k == 1 else "Top-5: true video in the 5 best guesses")
        ax.set_ylim(0, 60 if k == 5 else 23)
        ax.set_xlim(-0.6, len(steps) - 0.4)
        chance_line(ax, chance * 100)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_ENC, C_DEC, C_ENS)] + [
        plt.Line2D([0], [0], color=MUTED, ls="--", lw=1.6)]
    fig.legend(handles, ["Encoding (video → predicted EEG)", "Decoding (EEG → predicted video features)",
                         "Ensemble of two decoders", "Chance"], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08),
               fontsize=13)
    fig.suptitle("Each idea moved the score: from 2% to 15% top-1 (chance is 1%)", fontsize=21, fontweight="bold", y=1.03)
    fig.text(0.5, -0.11, "Development subjects 1–4, 408 recordings, 102 candidate videos each. Whiskers: 95% bootstrap "
             "interval over recordings (not available for the baseline).", ha="center", fontsize=12, color=MUTED)
    fig.tight_layout()
    save(fig, "fig1_progress")


# ---------------------------------------------------------------- figure 2: encoding vs decoding schematic
def _box(ax, xy, w, h, text, color, fs=15, tc="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="none", zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold",
            zorder=3, linespacing=1.25)


def _arrow(ax, a, b, color, rad=0.0, lw=3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=26, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def fig2_schematic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    for ax in axes:
        ax.set_xlim(0, 10), ax.set_ylim(0, 6), ax.axis("off"), ax.grid(False)
    # left: encoding
    ax = axes[0]
    ax.set_title("Encoding: predict the brain", color=C_ENC)
    _box(ax, (0.2, 4.1), 2.9, 1.4, "Video\n(CLIP features)", "#4D4D4D")
    _box(ax, (6.9, 4.1), 2.9, 1.4, "Predicted\nEEG", C_ENC)
    _box(ax, (6.9, 0.9), 2.9, 1.4, "Measured\nEEG", "#7A7A7A")
    _arrow(ax, (3.2, 4.8), (6.8, 4.8), C_ENC)
    ax.text(5, 5.25, "learned (ridge)", ha="center", color=C_ENC, fontsize=14)
    _arrow(ax, (8.35, 3.95), (8.35, 2.45), INK)
    _arrow(ax, (8.35, 2.45), (8.35, 3.95), INK)
    ax.text(8.55, 3.2, "compare", ha="left", va="center", fontsize=14)
    ax.text(5, 0.25, "Comparison happens in EEG space:\n17 channels × many time points, mostly noise",
            ha="center", fontsize=14.5, color=C_ENC, va="center")
    ax.text(0.2, 2.4, "5.4% top-1", fontsize=22, fontweight="bold", color=C_ENC)
    # right: decoding
    ax = axes[1]
    ax.set_title("Decoding: read the brain", color=C_DEC)
    _box(ax, (0.2, 4.1), 2.9, 1.4, "Measured\nEEG", "#7A7A7A")
    _box(ax, (6.9, 4.1), 2.9, 1.4, "Predicted\nCLIP features", C_DEC, tc=INK)
    _box(ax, (6.9, 0.9), 2.9, 1.4, "Video\n(CLIP features)", "#4D4D4D")
    _arrow(ax, (3.2, 4.8), (6.8, 4.8), C_DEC)
    ax.text(5, 5.25, "learned (ridge)", ha="center", color="#9A6A00", fontsize=14)
    _arrow(ax, (8.35, 3.95), (8.35, 2.45), INK)
    _arrow(ax, (8.35, 2.45), (8.35, 3.95), INK)
    ax.text(8.55, 3.2, "compare", ha="left", va="center", fontsize=14)
    ax.text(5, 0.25, "Comparison happens in feature space:\nsemantic gist of the scene, far less noise",
            ha="center", fontsize=14.5, color="#9A6A00", va="center")
    ax.text(0.2, 2.4, "14.0% top-1", fontsize=22, fontweight="bold", color="#9A6A00")
    fig.suptitle("Same data, opposite direction: comparing in meaning space wins", fontsize=21, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig2_encoding_vs_decoding")


# ---------------------------------------------------------------- figure 3: the whitening bug
def fig3_whitening() -> None:
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from eegmr.features import load_features

    _, ids = load_train(1)
    F = load_features("clip_b32", 8, ids, Path("features"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    spectra = {}
    for name, second in [("without", False), ("with", True)]:
        steps = [StandardScaler(), PCA(n_components=0.95, svd_solver="full")] + ([StandardScaler()] if second else [])
        Z = make_pipeline(*steps).fit_transform(F)
        spectra[name] = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    ax = axes[0]
    for name, color, label in [("without", C_ENC, "Fixed: scaler → PCA → ridge"),
                               ("with", C_DEC, "Bug: scaler → PCA → scaler → ridge")]:
        s2 = spectra[name]
        ax.plot(np.arange(1, len(s2) + 1), s2, color=color, lw=3.5, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Feature direction (strongest → weakest)")
    ax.set_ylabel("Strength of that direction (s², log scale)")
    ax.set_title("What ridge regression sees")
    ax.legend(loc="lower left")
    ax.annotate("all directions identical:\nridge shrinks them all equally", xy=(120, spectra["with"][120]), xytext=(70, 4000),
                fontsize=13.5, color="#9A6A00", arrowprops=dict(arrowstyle="->", color="#9A6A00"))
    # right: the consequence, dev top-1/top-5 from MODEL_BACKLOG M1 vs M2
    ax = axes[1]
    vals = {"top-1": (4.2, 5.4), "top-5": (16.9, 21.1)}
    x = np.arange(2)
    for i, (k, (bug, fixed)) in enumerate(vals.items()):
        ax.bar(x[i] - 0.19, bug, 0.36, color=C_DEC, zorder=3, label="Bug" if i == 0 else None)
        ax.bar(x[i] + 0.19, fixed, 0.36, color=C_ENC, zorder=3, label="Fixed" if i == 0 else None)
        ax.text(x[i] - 0.19, bug + 0.5 + (1.2 if k == "top-1" else 0), f"{bug}%", ha="center", fontsize=14, fontweight="bold")
        ax.text(x[i] + 0.19, fixed + 0.5, f"{fixed}%", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x, list(vals))
    ax.set_ylabel("Dev accuracy (%)")
    ax.set_ylim(0, 27)
    ax.set_title("Fixing it helps (p = 2e-13)")
    ax.text(1.5, CHANCE_TOP5 * 100 + 0.5, "chance", color=MUTED, fontsize=13, ha="center")
    ax.legend(loc="upper left")
    chance_line(ax, CHANCE_TOP5 * 100)
    fig.suptitle("A hidden bug made the regularization knob do nothing", fontsize=21, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_whitening_bug")


# ---------------------------------------------------------------- figure 4: when is the information in the EEG
def fig4_time() -> None:
    from eegmr.features import load_features
    from eegmr.gpu import decoding_cv_scores

    subjects = range(1, 7)
    starts = list(range(0, 141, 5))
    cache = DATA / "time_course.npz"  # ~9 min of GPU time; delete the file to recompute
    if cache.exists():
        with np.load(cache) as c:
            erps, scores = list(c["erps"]), list(c["scores"])
    else:
        win = 10  # 100 ms at 100 Hz
        alphas = np.logspace(1, 6, 11)
        erps, scores = [], []
        for s in subjects:
            X, ids = load_train(s)
            eeg = decimate(X[..., :750])  # first 1.5 s at 100 Hz
            erps.append(eeg.mean(0))
            F = load_features("clip_b32", 8, ids, Path("features"))
            cv = decoding_cv_scores(eeg, F, alphas, [slice(a, a + win) for a in starts])
            scores.append(0.5 - cv.min(axis=1))  # best alpha per window; > 0 means better than chance
            print("sub", s, "done", flush=True)
        np.savez(cache, erps=np.array(erps), scores=np.array(scores))
    erp = np.mean(erps, axis=0)  # (17, 150)
    t = np.arange(150) * 10
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    for ch in erp:
        ax.plot(t, ch, color=C_ENC, lw=1, alpha=0.45)
    gfp = np.sqrt((erp ** 2).mean(0))
    ax.plot(t, gfp, color=INK, lw=3.2, label="overall strength (RMS over 17 channels)")
    ax.set_ylabel("EEG amplitude (a.u.)")
    ax.set_title("Average EEG over all training videos (mean of 6 subjects)")
    ax.legend(loc="upper right")
    ax = axes[1]
    sc = np.array(scores)
    for s in sc:
        ax.plot(np.array(starts) * 10 + 50, s, color=C_DEC, lw=1, alpha=0.4)
    ax.plot(np.array(starts) * 10 + 50, sc.mean(0), color=INK, lw=3.2, label="mean of 6 subjects")
    ax.axhline(0, color=MUTED, ls="--", lw=1.6)
    ax.text(1480, 0.003, "chance", color=MUTED, ha="right", fontsize=13)
    ax.set_ylabel("Decoding skill\n(0 = chance; higher = better)")
    ax.set_xlabel("Time after video onset (ms)")
    ax.set_title("Which moments identify the video? (100 ms windows)")
    ax.legend(loc="upper right")
    for ax in axes:
        ax.axvspan(50, 600, color=C_ENS, alpha=0.13, zorder=0)
        ax.set_xlim(0, 1500)
    axes[0].text(325, axes[0].get_ylim()[1] * 0.93, "window used\nfor matching\n(50–600 ms)", ha="center", va="top",
                 fontsize=13.5, color="#00694B")
    fig.tight_layout()
    save(fig, "fig4_time_course")


# ---------------------------------------------------------------- figure 5: per subject
def fig5_subjects() -> None:
    cv = pd.read_csv(DATA / "ens_clip_patch_cv.csv")
    cv = cv[cv.weights == "0.5/0.5"].set_index("subject")  # equal weights = the final ensemble
    cv = cv.loc[[s for s in cv.index if str(s).isdigit()]]
    cv.index = cv.index.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    w = 0.38
    dev = {s: true_ranks(similarity("ens", s), load_test_labels(s)) for s in DEV_SUBJECTS}
    xs = np.array(DEV_SUBJECTS)
    for off, k, label, col in [(-w / 2, 1, "top-1", C_ENS), (w / 2, 5, "top-5", "#56B4E9")]:
        vals = [(dev[s] <= k).mean() * 100 for s in xs]
        ax.bar(xs + off, vals, w, color=col, zorder=3, label=label)
        for xi, v in zip(xs, vals):
            ax.text(xi + off, v + 0.8, f"{v:.0f}", ha="center", fontsize=13.5, fontweight="bold")
    ax.set_xticks(xs, [f"sub-{s:02d}" for s in xs])
    ax.set_ylabel("Correct video in top-k (%)")
    ax.set_title("Test accuracy, subjects with labels")
    ax.set_ylim(0, 58)
    chance_line(ax, CHANCE_TOP5 * 100)
    ax.legend(loc="upper right", ncol=2)
    ax.text(0.5, 6.2, "chance", color=MUTED, fontsize=13)
    ax = axes[1]
    x = np.arange(1, 7)
    ax.bar(x, cv.loc[x, "pct_rank"], color=C_ENS, zorder=3, label="Training-set cross-validation")
    for xi in x:
        ax.text(xi, cv.loc[xi, "pct_rank"] + 0.012, f"{cv.loc[xi, 'pct_rank']:.2f}", ha="center", fontsize=13,
                fontweight="bold")
    test_pr = [np.mean((dev[s] - 1) / (N_CAND - 1)) for s in DEV_SUBJECTS]
    ax.plot(DEV_SUBJECTS, test_pr, "D", color=INK, ms=11, zorder=5, label="Real test set")
    ax.set_xticks(x, [f"sub-{s:02d}" + ("\n(no labels)" if s > 4 else "") for s in x], fontsize=13)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Typical position of true video\n(0 = always 1st, 0.5 = chance)")
    ax.set_title("Average rank, all 6 subjects")
    chance_line(ax, 0.5)
    ax.text(0.6, 0.465, "chance", color=MUTED, fontsize=13)
    ax.legend(loc="upper center", fontsize=12.5, ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Consistent across subjects, with sub-05 and sub-06 expected in range", fontsize=21,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig5_per_subject")


# ---------------------------------------------------------------- figure 6: recall@k
def fig6_recall() -> None:
    ks = np.arange(1, N_CAND + 1)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    for model, label, color, ls in [("ens", "CLIP + DINOv2 decoders (final)", C_ENS, "-"),
                                    ("dec", "CLIP decoder", C_DEC, "-"), ("enc_noise", "Encoder + noise-aware matching", C_ENC, "-"),
                                    ("enc", "CLIP encoder", C_ENC, ":")]:
        r = dev_ranks(model)
        ax.plot(ks, [(r <= k).mean() * 100 for k in ks], color=color, lw=3.4 if model == "ens" else 2.4, ls=ls, label=label)
    ax.plot(ks, ks / N_CAND * 100, color=MUTED, ls="--", lw=2, label="chance")
    r = dev_ranks("ens")
    for k in (1, 5, 10, 20):
        ax.plot(k, (r <= k).mean() * 100, "o", color=C_ENS, ms=9, zorder=5)
        ax.annotate(f"{(r <= k).mean() * 100:.0f}%", (k, (r <= k).mean() * 100), xytext=(12, -30), textcoords="offset points",
                    fontsize=14, fontweight="bold", color="#00694B",
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel("k: number of top guesses allowed")
    ax.set_ylabel("True video among the top k (%)")
    ax.set_title("Where does the true video land? (out of 102)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig6_recall_at_k")


# ---------------------------------------------------------------- figure 7: failure cases
def _frame(video_id: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path(int(video_id))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, img = cap.read()
    cap.release()
    assert ok, f"cannot read video {video_id}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fig7_examples(picks: list[tuple[int, int]] | None = None) -> None:
    """Rows: (subject, recording index). Default: one hit and two near misses from sub-01, fixed by the rule below."""
    S = similarity("ens", 1)
    ids = load_test_labels(1)
    ranks = true_ranks(S, ids)
    if picks is None:
        chosen = [int(np.flatnonzero(ranks == r)[j]) for r, j in [(1, 0), (2, 0), (4, 0)]]
        picks = [(1, i) for i in chosen]
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.5 * len(picks)))
    for row, (sub, i) in enumerate(picks):
        S, ids = similarity("ens", sub), load_test_labels(sub)
        order = np.argsort(-S[i])
        top3 = TEST_IDS[order[:3]]
        rank = int(true_ranks(S, ids)[i])
        shown = [(ids[i], "True video", C_ENS)] + [
            (v, f"Guess #{j + 1}" + ("  ✓ correct" if v == ids[i] else ""), C_ENS if v == ids[i] else MUTED)
            for j, v in enumerate(top3)]
        for col, (vid, title, color) in enumerate(shown):
            ax = axes[row, col]
            ax.imshow(_frame(vid)), ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True), sp.set_color(color), sp.set_linewidth(5 if color == C_ENS else 1.5)
            ax.set_title(title if col else f"{title} (ranked #{rank})", fontsize=15, color=INK if col == 0 else color)
    fig.suptitle("One hit and two near misses (sub-01, final model)", fontsize=20, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig7_examples")


# ---------------------------------------------------------------- figure 8: feature comparison
def fig8_features() -> None:
    rows = [  # label, model key (None = AlexNet), colour, head
        ("AlexNet", None, C_ENC), ("CLIP", "enc", C_ENC),
        ("CLIP", "decw", C_DEC), ("DINOv2\n(summary)", "dino_cls", C_DINO),
        ("DINOv2\n(patches)", "dino_patch", C_DINO), ("CLIP + DINOv2", "ens", C_ENS),
    ]
    fig, ax = plt.subplots(figsize=(14, 6.4))
    x = np.array([0, 1, 2.8, 3.8, 4.8, 5.8])
    for xi, (label, model, color) in zip(x, rows):
        if model is None:
            val = alexnet_dev()["top5"].mean()
            ax.bar(xi, val * 100, 0.78, color=color, zorder=3)
            top = val
        else:
            r = dev_ranks(model)
            val = (r <= 5).mean()
            lo, hi = boot_ci(r, 5)
            ax.bar(xi, val * 100, 0.78, color=color, zorder=3)
            ax.errorbar(xi, val * 100, yerr=[[(val - lo) * 100], [(hi - val) * 100]], color=INK, capsize=4, lw=1.6, zorder=4)
            top = hi
        ax.text(xi, top * 100 + 1, f"{val * 100:.0f}%", ha="center", fontsize=15, fontweight="bold")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=12.5)
    ax.set_xlim(-0.7, 6.5)
    ax.set_ylim(0, 56)
    ax.set_ylabel("True video in top 5 (%)")
    ax.axvline(2.15, color=GRID, lw=2)
    ax.text(0.5, 53, "Encoding head", ha="center", fontsize=15, color=C_ENC, fontweight="bold")
    ax.text(4.3, 53, "Decoding head (noise-whitened)", ha="center", fontsize=15, color="#9A6A00", fontweight="bold")
    chance_line(ax, CHANCE_TOP5 * 100)
    ax.text(1.85, CHANCE_TOP5 * 100 + 0.8, "chance", color=MUTED, fontsize=13, ha="center")
    ax.set_title("Which visual features? Combining two helps most")
    fig.text(0.5, -0.03, "DINOv2 features were only tested in the decoding head (left group uses the encoding head). Dev subjects "
             "1–4; whiskers: 95% bootstrap interval.", ha="center", fontsize=12, color=MUTED)
    fig.tight_layout()
    save(fig, "fig8_features")


FIGS = {"fig1": fig1_progress, "fig2": fig2_schematic, "fig3": fig3_whitening, "fig4": fig4_time, "fig5": fig5_subjects,
        "fig6": fig6_recall, "fig7": fig7_examples, "fig8": fig8_features}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
