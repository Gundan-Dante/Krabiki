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
    "v3": "v3_headline", "v4": "v4_headline", "final": "final_csls",
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
C_V3 = "#005C44"


def fig1_progress() -> None:
    """Dev (sub-01..04) top-1 / top-5 in %, numbers from MODEL_BACKLOG / PROGRESS (M-ids in the labels)."""
    steps = [  # label, top-1, top-5, colour, hatch
        ("M0\nAlexNet\nbaseline", 2.0, 8.6, C_ENC, None),
        ("M2\nCLIP\nencoding", 5.4, 21.1, C_ENC, None),
        ("M5\n+ noise-\naware match", 8.8, 31.6, C_ENC, None),
        ("M3\nCLIP\ndecoding", 14.0, 36.0, C_DEC, None),
        ("M8a\n+ whitened\ninput", 14.7, 42.2, C_DEC, None),
        ("M11\n+ DINOv2", 15.2, 45.3, C_ENS, None),
        ("M13\n+ contrastive\nhead", 17.4, 46.8, C_ENS, None),
        ("M17\n+ V-JEPA\nvideo model", 20.8, 48.0, C_ENS, None),
        ("final_csls\n+ fixes,\nCSLS", 24.0, 57.1, C_ENS, "///"),
        ("v3\n+ sound,\nlow-level", 32.4, 66.7, C_V3, "///"),
        ("v4 headline\n+ subject\npooling", 33.3, 68.9, C_V3, "///"),
    ]
    n_trials = 408
    fig, axes = plt.subplots(1, 2, figsize=(24, 7.4))
    for ax, (key, chance, ymax) in zip(axes, [("top-1: exact match", CHANCE_TOP1, 40), ("top-5: true video in the 5 best guesses", CHANCE_TOP5, 80)]):
        j = 1 if key.startswith("top-1") else 2
        for i, st in enumerate(steps):
            if st[1] is None:  # placeholder
                ax.add_patch(plt.Rectangle((i - 0.36, 0), 0.72, ymax * 0.12, fill=False, ls="--", ec=MUTED, lw=2, zorder=3))
                ax.text(i, ymax * 0.15, "TBD", ha="center", va="bottom", color=MUTED, fontsize=15, fontweight="bold")
                continue
            v = st[j]
            ax.bar(i, v, 0.72, color=st[3], hatch=st[4], edgecolor="white" if st[4] else st[3], lw=0, zorder=3)
            se = 1.96 * np.sqrt(v / 100 * (1 - v / 100) / n_trials) * 100
            ax.errorbar(i, v, yerr=se, color=INK, capsize=4, lw=1.6, zorder=4)
            ax.text(i, v + se + ymax * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(steps)), [st[0] for st in steps], fontsize=10)
        ax.set_xlim(-0.6, len(steps) - 0.4)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"Correct video in top {1 if j == 1 else 5} (%)")
        ax.set_title(key)
        chance_line(ax, chance * 100)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_ENC, C_DEC, C_ENS, C_V3)] + [
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, hatch="///"), plt.Line2D([0], [0], color=MUTED, ls="--", lw=1.6)]
    fig.legend(handles, ["Encoding (video \u2192 predicted EEG)", "Decoding (EEG \u2192 video features)", "Ensemble of decoders",
                         "Ensemble + sound and low-level members", "Hatched: with centering + CSLS", "Chance"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.11), fontsize=13)
    fig.suptitle("Each idea moved the score: from 2% to 33% top-1 (chance is 1%)", fontsize=22, fontweight="bold", y=1.02)
    fig.text(0.5, -0.17, "Development subjects 1\u20134, 408 recordings, 102 candidate videos each. Whiskers: approximate 95% interval (binomial, 408 recordings).",
             ha="center", fontsize=12, color=MUTED)
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
    save(fig, "fig3_time_course")


# ---------------------------------------------------------------- figure 6: recall@k
def fig6_recall() -> None:
    ks = np.arange(1, N_CAND + 1)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    for model, label, color, ls, lw in [("v4", "v4 ensemble (headline)", C_V3, "-", 3.6), ("v3", "v3 ensemble", C_ENS, "--", 2.4), ("final", "final_csls (previous headline)", C_ENS, "-", 2.4),
                                        ("decw", "CLIP decoder alone (M8a)", C_DEC, "-", 2.4), ("enc_noise", "Encoder + noise-aware matching (M5)", C_ENC, "-", 2.4)]:
        r = dev_ranks(model)
        ax.plot(ks, [(r <= k).mean() * 100 for k in ks], color=color, lw=lw, ls=ls, label=label)
    ax.plot(ks, ks / N_CAND * 100, color=MUTED, ls="--", lw=2, label="chance")
    r = dev_ranks("v4")
    for k in (1, 5, 10, 20):
        ax.plot(k, (r <= k).mean() * 100, "o", color=C_V3, ms=9, zorder=5)
        ax.annotate(f"{(r <= k).mean() * 100:.0f}%", (k, (r <= k).mean() * 100), xytext=(12, -30), textcoords="offset points",
                    fontsize=14, fontweight="bold", color=C_V3, bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel("k: number of top guesses allowed")
    ax.set_ylabel("True video among the top k (%)")
    ax.set_title("Where does the true video land? (out of 102)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig7_recall_at_k")


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
    S = similarity("v4", 1)
    ids = load_test_labels(1)
    ranks = true_ranks(S, ids)
    if picks is None:
        chosen = [int(np.flatnonzero(ranks == r)[j]) for r, j in [(1, 0), (2, 0), (4, 0)]]
        picks = [(1, i) for i in chosen]
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.5 * len(picks)))
    for row, (sub, i) in enumerate(picks):
        S, ids = similarity("v4", sub), load_test_labels(sub)
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
    fig.suptitle("One hit and two near misses (sub-01, v4 model)", fontsize=20, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig8_examples")


# ---------------------------------------------------------------- figure 4: ensemble anatomy
MEMBERS = ["DINOv2\n(image)", "V-JEPA\n(video)", "CLIP\ntwo-tower", "AlexNet\nmid-level", "Hand-made\ndescriptors", "CLAP\n(sound)", "Low-level\nsound+image"]
# PROGRESS.md "FINAL v3": per-subject CV-chosen weights (patch / V-JEPA / con / conv5 / descriptors / CLAP / lowav)
V3_WEIGHTS = {1: [.2, .1, .1, .2, .2, .1, .1], 2: [.1, .1, .1, .2, .2, .1, .2], 3: [.1, .2, .1, .2, .1, .1, .2],
              4: [.2, .1, .1, .1, .2, .1, .2], 5: [.2, .1, 0, .2, .2, 0, .3], 6: [.2, .1, 0, .2, .1, .1, .3]}
# PROGRESS.md "FINAL v3": drop-one ablation, LOFO change of the global strategy (lowav: that run was cut off)
V3_DROP = {0: -0.0073, 1: -0.0032, 2: -0.0046, 3: -0.0104, 4: -0.0106, 5: -0.0013}


def fig4_ensemble() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    # (a) weights
    ax = axes[0]
    W = np.array([V3_WEIGHTS[s] for s in range(1, 7)]).T  # members x subjects
    im = ax.imshow(W, cmap="Greens", vmin=0, vmax=0.45, aspect="auto")
    ax.grid(False)
    for i in range(7):
        for j in range(6):
            ax.text(j, i, f"{W[i, j]:.1f}", ha="center", va="center", fontsize=14, fontweight="bold" if W[i, j] >= 0.2 else None,
                    color="white" if W[i, j] >= 0.3 else INK)
    ax.set_xticks(range(6), [f"sub-{s:02d}" for s in range(1, 7)], fontsize=12)
    ax.set_yticks(range(7), MEMBERS, fontsize=12)
    ax.set_title("(a) How much each member counts")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.annotate("sub-05/06:\nmore weight on\nlow-level sound+image", xy=(4.5, 6), xytext=(4.5, 7.55), ha="center", va="top", fontsize=11.5, color="#00694B",
                annotation_clip=False)
    # (b) overlap
    ax = axes[1]
    M = np.zeros((7, 7))
    z = lambda S: (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)  # noqa: E731
    for s in range(1, 7):
        X = [z(np.load(DATA / "sim" / f"v3_member{i}" / f"sub-{s:02d}.npy")).ravel() for i in range(7)]
        M += np.corrcoef(X) / 6
    mask = np.triu(np.ones_like(M, dtype=bool), 1)
    Mm = np.where(mask, np.nan, M)
    ax.imshow(Mm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(7):
        for j in range(7):
            if j <= i and i != j:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=13, color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(7), MEMBERS, fontsize=10.5, rotation=0)
    ax.set_yticks(range(7), MEMBERS, fontsize=10.5)
    ax.set_title("(b) How alike the members' scores are")
    for sp in ax.spines.values():
        sp.set_visible(False)
    # (c) drop-one
    ax = axes[2]
    y = np.arange(7)[::-1]
    for i in range(7):
        if i in V3_DROP:
            ax.barh(y[i], -V3_DROP[i], color=C_ENS, height=0.65, zorder=3)
            ax.text(-V3_DROP[i] + 0.0002, y[i], f"{-V3_DROP[i]:.4f}", va="center", fontsize=12.5)
        else:
            ax.text(0.0002, y[i], "not measured (run cut off)", va="center", fontsize=11.5, color=MUTED)
    ax.set_yticks(y, MEMBERS, fontsize=12)
    ax.set_ylim(-0.7, 6.7)
    ax.set_xlim(0, 0.0135)
    ax.set_xlabel("Score lost when removed (higher = matters more)", fontsize=14)
    ax.set_title("(c) Remove one member")
    fig.suptitle("Diverse views of the video, and each one earns its place", fontsize=22, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03, "v3 ensemble. (a) CV-chosen weights per subject (sum = 1). (b) Correlation of the members' test similarity scores, mean over 6 subjects, recomputed from saved "
             "matrices (within 0.02 of M27). (c) Change in leave-one-fold-out CV (training data only) of the 7-member ensemble.", ha="center", fontsize=12, color=MUTED)
    fig.tight_layout()
    save(fig, "fig4_ensemble_anatomy")


# ---------------------------------------------------------------- figure 5: expected score on the scored subjects
def fig5_expected() -> None:
    cv = pd.read_csv(DATA / "v4_cv_selected.csv").set_index("subject")
    dev = {s: true_ranks(similarity("v4", s), load_test_labels(s)) for s in DEV_SUBJECTS}
    d1 = {s: (dev[s] <= 1).mean() * 100 for s in dev}
    d5 = {s: (dev[s] <= 5).mean() * 100 for s in dev}
    e1, e5 = cv.exp_top1 * 100, cv.exp_top5 * 100
    r1 = [d1[s] / e1[s] for s in dev]
    r5 = [d5[s] / e5[s] for s in dev]
    proj = {}
    for s in (5, 6):
        proj[s] = {1: (e1[s] * min(r1), e1[s] * np.mean(r1), e1[s] * max(r1)), 5: (e5[s] * min(r5), e5[s] * np.mean(r5), e5[s] * max(r5))}
        print(f"sub-{s:02d} projected top-1 {proj[s][1][0]:.0f}-{proj[s][1][2]:.0f} (mid {proj[s][1][1]:.0f}), top-5 {proj[s][5][0]:.0f}-{proj[s][5][2]:.0f} (mid {proj[s][5][1]:.0f})")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6))
    for ax, k, dv, ev, col, title in [(axes[0], 1, d1, e1, C_ENS, "Top-1: exact match"), (axes[1], 5, d5, e5, "#56B4E9", "Top-5: true video in the 5 best guesses")]:
        x = np.arange(1, 7)
        for s in x:
            if s in dv:
                ax.bar(s, dv[s], 0.62, color=col, zorder=3, label="Real test score" if s == 1 else None)
                ax.text(s, dv[s] + 1, f"{dv[s]:.0f}", ha="center", fontsize=13.5, fontweight="bold")
            else:
                lo, mid, hi = proj[s][k]
                ax.bar(s, mid, 0.62, fc="none", ec=col, hatch="//", lw=2, zorder=3, label="Projected (estimate)" if s == 5 else None)
                ax.errorbar(s, mid, yerr=[[mid - lo], [hi - mid]], color=INK, capsize=7, lw=2.2, zorder=5)
                ax.text(s, hi + 1, f"~{mid:.0f}", ha="center", fontsize=13.5, fontweight="bold")
        ax.plot(x, [ev[s] for s in x], "D", color=INK, ms=10, zorder=6, label="Estimate from training data (cross-validation)")
        ax.set_xticks(x, [f"sub-{s:02d}" + ("\n(scored)" if s > 4 else "") for s in x], fontsize=13)
        ax.set_ylabel(f"Correct video in top {k} (%)")
        ax.set_title(title)
        ax.set_ylim(0, 85 if k == 5 else 52)
        chance_line(ax, (CHANCE_TOP1 if k == 1 else CHANCE_TOP5) * 100)
    axes[0].legend(loc="upper right", fontsize=12)
    fig.suptitle("What to expect on the two scored subjects: an estimate, not a measurement", fontsize=21, fontweight="bold", y=1.03)
    fig.text(0.5, -0.07, "Projection = cross-validation estimate \u00d7 the test/cross-validation ratio seen on sub-01..04 (whisker: lowest to highest ratio). sub-05 and sub-06 are the weakest "
             "subjects by cross-validation, and the dev numbers were selected among many variants, so treat these as optimistic.", ha="center", fontsize=12, color=MUTED, wrap=True)
    fig.tight_layout()
    save(fig, "fig5_expected_score")


# ---------------------------------------------------------------- figure 6: subject pooling
# MODEL_BACKLOG M26: DINOv2-patch decoder, train-CV (E[top-1] + E[top-5]) / 2, own data only (lambda = 0) vs best pooled lambda
POOL = {1: (0.213, 0.220, 0.1), 2: (0.186, 0.195, 0.05), 3: (0.149, 0.159, 0.1), 4: (0.180, 0.187, 0.1), 5: (0.145, 0.151, 0.05), 6: (0.118, 0.135, 0.2)}


def fig6_pooling() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    for s, (a, b, lam) in POOL.items():
        ax.plot([s, s], [a, b], color=MUTED, lw=2.5, zorder=2)
        ax.plot(s, a, "o", color=C_DEC, ms=13, zorder=3, label="Own recordings only" if s == 1 else None)
        ax.plot(s, b, "o", color=C_ENS, ms=13, zorder=3, label="+ other subjects' recordings (small weight)" if s == 1 else None)
        ax.text(s + 0.12, (a + b) / 2, f"+{(b / a - 1) * 100:.0f}%\nweight {lam:g}", fontsize=12.5, va="center", fontweight="bold" if s == 6 else None,
                color="#00694B" if s == 6 else INK)
    ax.set_xticks(range(1, 7), [f"sub-{s:02d}" for s in range(1, 7)])
    ax.set_xlim(0.6, 6.9)
    ax.set_ylim(0.10, 0.24)
    ax.set_ylabel("Cross-validation score (higher = better)")
    ax.set_title("Borrowing a little data from other subjects helps the weakest most")
    ax.legend(loc="upper right", fontsize=13)
    fig.text(0.5, -0.04, "Single DINOv2 decoder, training-set cross-validation, mean of expected top-1 and top-5. Test-set effect (sub-01..04): 12.7 / 41.7 \u2192 13.0 / 41.7, not significant.",
             ha="center", fontsize=12, color=MUTED)
    fig.tight_layout()
    save(fig, "fig6_subject_pooling")


# ---------------------------------------------------------------- figure 9: diagnostics (data-limited, moving videos)
def fig9_diagnostics() -> None:
    df = pd.read_csv(DATA / "stage3_curve_clip_b32.csv")
    fr = [0.5, 0.75, 1.0]
    ridge = df[df.config == "ridge"].groupby("train_fraction").cv_score.mean().reindex(fr)
    eeg = df[df.config.str.startswith("eegnet")].groupby("train_fraction").cv_score.mean().reindex(fr)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    ax = axes[0]
    ax.plot([f * 100 for f in fr], ridge, "o-", color=C_DEC, lw=3, ms=10, label="Ridge decoder (what we use)")
    ax.plot([f * 100 for f in fr], eeg, "o-", color=C_DINO, lw=3, ms=10, label="Small neural network (EEGNet)")
    for f in fr:
        ax.text(f * 100 + 1.5, (eeg[f] + ridge[f]) / 2, f"gap {eeg[f] - ridge[f]:.3f}", ha="left", va="center", fontsize=12.5)
    ax.set_xticks([50, 75, 100], ["50%", "75%", "100%"])
    ax.set_xlim(45, 112)
    ax.set_ylim(0.22, 0.375)
    ax.set_xlabel("Share of the training recordings used")
    ax.set_ylabel("Typical position of true video\n(lower = better, 0.5 = chance)")
    ax.set_title("More data closes the gap to the neural network")
    ax.legend(loc="upper right")
    ax = axes[1]
    # MODEL_BACKLOG M23 (krabiki-2f): blend decoding score of the true video by motion tercile, 5,650 held-out training trials
    vals = [0.179, 0.178, 0.201]
    ax.bar(range(3), vals, 0.6, color=[C_ENS, C_ENS, C_DEC], zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(range(3), ["Static\nvideos", "Medium", "Moving\nvideos"])
    ax.set_ylim(0, 0.26)
    ax.set_ylabel("Typical position of true video\n(lower = better, 0.5 = chance)")
    ax.set_title("Moving videos are slightly harder")
    fig.suptitle("Two diagnostics", fontsize=21, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "Left: training-set cross-validation, mean of 6 subjects, CLIP target (M14). Right: held-out training trials, 6 subjects (M23); moving minus static = +0.018, "
             "95% interval +0.006 to +0.031.", ha="center", fontsize=12, color=MUTED)
    fig.tight_layout()
    save(fig, "fig9_diagnostics")


FIGS = {"fig1": fig1_progress, "fig2": fig2_schematic, "fig3": fig4_time, "fig4": fig4_ensemble, "fig5": fig5_expected,
        "fig6": fig6_pooling, "fig7": fig6_recall, "fig8": fig7_examples, "fig9": fig9_diagnostics}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
