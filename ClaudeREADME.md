# EEG Mind Reading: Build Brief

This is a build brief for an AI coding assistant (or a teammate). It describes the challenge, the data, the rules, and the system to build. **Read all of it before writing any code.**

Hackathon: National AI Hackathon 2026 (SchubKI), challenge "Mind Reading Using Machine Learning and EEG" (Constructor Technology). Work days: Oct 17–18, 2026.

## 0. How this track fits in (read first)

This brief is **one of several parallel tracks**. It covers improving the proven, published approach. Other team members are researching completely different solutions at the same time. So:

- **Work independently.** Build everything in its own folder or branch (for example `track-proven/`). Do not modify the organizers' notebooks or other people's code.
- **Shared parts must be reusable by other tracks.** The submission writer, the evaluation harness (Stage 0), and the cached video/audio features (Stage 1) are needed by every approach. Keep them clean, documented, and importable on their own.
- **Deliver value after every stage.** After each stage there must be a working `predict.py` that produces valid submission files with the best model so far. If we stop at any point, we can still submit.
- **Report as you go.** Append every experiment to `results/results.csv`. Keep a short `PROGRESS.md` covering what's done, the current best top-1/top-5 on sub-01..04, and what's next. The team checks it periodically.
- **Use one comparison standard.** Use the same evaluation function and the same dev subjects (sub-01..04) for all numbers, so results from different tracks can be compared directly.

---

## 1. The task

People watched short (~3 s) videos with sound while EEG was recorded. Given one EEG recording and the 102 candidate test videos, **rank the 102 videos from most to least likely** to have caused that brain activity.

- Scoring (70% of the grade): **top-1** and **top-5 accuracy** on the test recordings of **sub-05 and sub-06**. Chance is 1/102 ≈ 1.0% for top-1 and 5/102 ≈ 4.9% for top-5.
- The other 30%: technical soundness and novelty (15%), code quality and reproducibility (10%), and the presentation, including failure-case analysis (5%).

**Key fact:** the 102 test videos (IDs 1001–1102) are **never** in the training set. So this is **zero-shot retrieval**: we cannot learn "EEG → video ID". We have to map EEG and videos into a space where they can be compared.

**Second key fact (easy to get wrong):** sub-05 and sub-06 are only *partially* held out. **Their training recordings and labels are provided** and may be used for training. Only their test labels are withheld. So every subject has its own labeled training data, and the four dev subjects (sub-01..04) mirror the sub-05/06 situation exactly.

---

## 2. Data

Root folder: `eeg-mind-reading-challenge/`. **Do not commit this folder.** Videos and extracted frames must not go into Git.

| Path | Content |
|---|---|
| `stimuli_train/0001.mp4 … 1000.mp4` | 1,000 training videos (with audio) |
| `stimuli_test/1001.mp4 … 1102.mp4` | 102 test videos, viewed by every subject |
| `trn/sub-XX_trn.npz` (XX = 01..06) | dict: key = video ID as a string **without leading zeros** (`"35"` → `0035.mp4`); value = float array `(17, 1500)` |
| `trn/sub-XX_trn/<id>.npy` | the same data unpacked into one file per ID (redundant copy; use the `.npz`) |
| `tst/sub-XX_tst.npy` | `(102, 17, 1500)`: one recording per test video, **in shuffled order** |
| `tst_stim/sub-XX_tst_stim.npy` | `(102,)` true video IDs (1001–1102) in the same order as `tst`. **Provided only for sub-01..04.** |
| `01_preprocess_and_train_model.ipynb`, `02_predict_movies.ipynb` | organizers' example baseline (reference only) |

EEG format: 17 channels, 500 Hz, 3 s from video onset (1500 samples). Each subject has 850–972 training recordings, and some video IDs are missing per subject, so handle the gaps. Channel names are not given; do not assume a montage.

Loading:
```python
trn = np.load("eeg-mind-reading-challenge/trn/sub-01_trn.npz")
ids = np.array(sorted(trn.files, key=int), dtype=int)
X = np.stack([trn[str(i)] for i in ids])        # (n, 17, 1500)
tst = np.load("eeg-mind-reading-challenge/tst/sub-01_tst.npy")   # (102, 17, 1500)
```

---

## 3. Rules (hard constraints)

1. **Python 3.11.**
2. External models and datasets must be **public and released before Oct 16, 2026**. Record the exact name and version of every pretrained model (for example `open_clip ViT-B-32 / laion2b_s34b_b79k`).
3. **Never train or fine-tune on test data** of any subject.
4. Test recordings and labels of sub-01..04 may be used **only for validation and model selection**. They must **not** be used as templates or reference responses for identifying sub-05/06 recordings.
5. **Unsupervised normalization of test recordings and of video features is allowed** (for example z-scoring).
6. Submit code, the training and inference pipeline, our own model weights, and the predictions. It must be reproducible.

---

## 4. Required output format (strict)

- Files: `sub-05_ranked_candidate_list.npy` and `sub-06_ranked_candidate_list.npy`, saved with `numpy.save`.
- Integer array, shape **(102, 102)**.
- **Row i = the i-th recording of `tst/sub-XX_tst.npy` in its ORIGINAL order.** Never sort or reorder the rows.
- Each row contains **all 102 video IDs (1001..1102)**, not column indices 0..101, ordered from most to least likely.
- Wrong name, shape, or content means the file cannot be scored.

**Known bug in the organizers' example `02_predict_movies.ipynb`. Do not copy it:**
- It reorders test EEG with `np.argsort(gt)` (the answer key). That only works for dev subjects and breaks the row order. sub-05/06 have no `gt`.
- The save line `np.save(f'sub-0{subject_num}...', eeg_arr)` uses undefined variables and saves the wrong array.
- `np.load(+FEAT_TEST_DIR+...)` has a stray `+`.

The submission code path must be **the same function** for dev and held-out subjects. Validate it by producing `sub-01_ranked_candidate_list.npy` and scoring it against `tst_stim/sub-01_tst_stim.npy`.

---

## 5. Baseline to beat (organizers' example)

Encoding model, trained per subject:
1. Decimate EEG from 500 to 100 Hz (`scipy.signal.decimate`, FIR, zero-phase).
2. 8 evenly spaced frames per video → AlexNet `features.3` (via `net2brain`) → mean over frames.
3. StandardScaler → PCA (95% variance) → StandardScaler → `RidgeCV` predicting the full EEG (17 × 300).
4. Retrieval: correlate measured test EEG with predicted EEG for each candidate over samples 5:60 (≈50–600 ms at 100 Hz), then rank.

Reported result: about 4× chance on test (top-1 ≈ 4%), and 67% top-1 on *training* data (overfit). The ERP shows peaks around 0.1 s (early visual, P1/N1) and 0.2–0.3 s (higher-level processing).

---

## 6. What to build

Build in **stages**. Each stage must end with a validated number on sub-01..04 before the next one starts. Keep the best working version runnable at all times.

### Stage 0: Evaluation harness and submission writer (do this FIRST)
- `evaluate(similarity: (102,102), true_ids) -> top1, top5, median_rank`
- `write_submission(similarity, subject, out_dir)`: rows in original order, values are video IDs, plus asserts for shape, dtype, and each row being a permutation of 1001..1102.
- A self-test: write the sub-01 file, reload it, and score it against `tst_stim`.

### Stage 1: Cached stimulus features (frozen, pretrained)
For all 1,102 videos, compute features once and cache them to `features/*.npy`. Frames and features stay outside Git.
- **Vision (semantic):** `open_clip` (ViT-B-32 to start; ViT-L-14 if time allows) on 8–16 evenly spaced frames. Save per-frame embeddings and the mean.
- **Vision (alternative):** DINOv2 (`facebook/dinov2-small` or `-base`).
- **Audio:** the videos have sound. Extract audio with ffmpeg and embed it with CLAP (`laion-clap`) or a similar public audio model. This is an underused signal and counts toward novelty.
- Optional: low-level features (mean luminance and contrast over time, frame-difference motion energy). These match early EEG responses well and are cheap.
- The laptop is **CPU only**, so keep models small and cache everything.

### Stage 2: Strong linear baseline (fast, robust)
Rerun the baseline recipe, but with CLIP/DINOv2/audio features instead of AlexNet. Two variants, both with ridge regression:
- **Encoding:** features → predicted EEG, then correlation matching (as in the baseline).
- **Decoding:** EEG (flattened, time-windowed) → feature space, then cosine similarity to candidate video features.

Improvements to try: per-subject z-scoring of EEG (per channel, using training statistics plus allowed unsupervised normalization of test data), a tuned time window, PCA on the EEG target, and banded ridge (separate regularization per feature type).

Data handling: train on the subject's own training data, and also try **pooling the training data of all 6 subjects** (after per-subject normalization). Report both.

### Stage 3: Contrastive EEG encoder (main model)
CLIP-style two-tower retrieval:
- **Video tower:** frozen cached features (vision + audio, concatenated or fused) → small trainable projection head.
- **EEG tower:** small network on (17 × T) input. Start with **EEGNet** (`braindecode`), then try a temporal-spatial CNN with channel attention (ATM-style). Keep it small: data is about 5,500 trials total.
- **Loss:** symmetric InfoNCE with a learnable temperature.
- **Subjects:** train on all 6 subjects' training data. Use per-subject normalization and a learned per-subject embedding or affine layer.
- **Regularization:** dropout, weight decay, early stopping on validation, augmentation (time shift ±20 ms, channel dropout, Gaussian noise, mixup between trials of the same video when repeats exist).
- **Validation split:** hold out a random 10% of training *videos* (not trials) for early stopping. Report final numbers on the sub-01..04 test sets.

### Stage 4: Ranking tricks and ensemble
- **CSLS** (cross-domain similarity local scaling) on the similarity matrix to reduce "hub" videos that rank high for everything.
- **Ensemble:** z-score each model's similarity matrix row-wise, then average them (linear + contrastive, vision + audio). Pick weights on sub-01..04 only.
- **Optional, check with organizers first:** each test video appears exactly once per subject, so a one-to-one assignment (Hungarian algorithm, or Sinkhorn-normalized similarities) could raise top-1. It uses no labels, but it exploits the test structure and may count as dataset hacking. Implement it behind a flag that is **off by default**, and report results with and without it.

---

## 7. Validation protocol

- Primary metric: mean top-1 and top-5 across sub-01..04 test sets. Also report per-subject numbers and the median rank.
- The model used for sub-05/06 must be trained exactly as for dev subjects (same code, same hyperparameters). Hyperparameters may be chosen only from dev results.
- Fix random seeds and report mean ± std over 3 seeds for the main models.
- Keep a results table (`results/results.csv`) with one row per experiment and its config.

---

## 8. Scientific analysis (for the report and slides)

Judges reward insight into how brains and models process video. Produce figures for:
1. **Time-resolved decoding:** accuracy using sliding EEG windows (for example 100 ms). When is stimulus information present?
2. **Channel importance:** accuracy when dropping each channel (or from model weights).
3. **Feature-space comparison:** low-level vs AlexNet vs CLIP vs DINOv2 vs audio. Which matches the EEG best, and at which latency?
4. **Failure cases:** which videos get confused with each other, and do the confusions share semantics (same category or scene)? Expectation from the literature: EEG carries coarse, gist-level information, so confusions should be within-category.
5. **Subject variability:** per-subject accuracy, and the effect of pooling subjects.

---

## 9. Repository layout

```
.
├── README.md                # how to install, train, predict, reproduce
├── requirements.txt         # pinned, Python 3.11
├── .gitignore               # data/, frames/, features/, *.mp4, large weights if needed
├── configs/                 # YAML configs per experiment
├── src/eegmr/
│   ├── data.py              # loading, decimation, normalization
│   ├── features.py          # frame/audio extraction, pretrained embeddings, caching
│   ├── linear.py            # ridge encoding/decoding models
│   ├── contrastive.py       # EEG encoder, projection heads, InfoNCE training
│   ├── retrieval.py         # similarity, CSLS, ensemble, (optional) assignment
│   ├── evaluate.py          # top-1/top-5, median rank
│   └── submission.py        # writes + validates ranked candidate lists
├── scripts/
│   ├── extract_features.py
│   ├── train.py             # --config configs/xxx.yaml
│   ├── predict.py           # writes sub-0X_ranked_candidate_list.npy
│   └── analyze.py           # figures for the report
├── weights/                 # our trained weights (small; commit them)
├── submissions/             # sub-05/06 ranked lists (commit them)
└── results/                 # results.csv, figures
```

Requirements:
- Every path is configurable, with the data root defaulting to `eeg-mind-reading-challenge/`.
- One command each to extract features, train, and predict.
- Deterministic seeds and logged configs.
- Type hints, docstrings, no notebook-only logic.

---

## 10. Environment

Ubuntu laptop, CPU only, Python 3.11 venv. Already installed:

```
torch torchvision (CPU), numpy, scipy, scikit-learn, pandas, matplotlib, joblib, tqdm,
mne, braindecode, opencv-python, open_clip_torch, einops,
net2brain (pip install -U git+https://github.com/cvai-roig-lab/Net2Brain)
```

Probably still needed: `ffmpeg` (`sudo apt install ffmpeg`), `laion-clap` or another audio model, `librosa`, `pyyaml`. Pin final versions in `requirements.txt`.

---

## 11. Priorities if time runs out

1. Correct, validated submission files (Stage 0). **Without these we score zero.**
2. Stage 2 with CLIP features (most likely large, cheap gain over the baseline).
3. CSLS and ensembling.
4. Contrastive model (Stage 3).
5. Audio features.
6. Analysis figures and the report.

---

## 12. References

- ATM: *Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion* (NeurIPS 2024). https://arxiv.org/abs/2403.07721
- NICE: *Decoding Natural Images from EEG for Object Recognition*. https://arxiv.org/abs/2308.13234
- NeuroCLIP (cross-subject EEG–image contrastive learning). https://arxiv.org/abs/2511.09250
- SAMGA: subject-aware alignment for zero-shot EEG retrieval. https://arxiv.org/abs/2604.17782
- CSLS and layer selection for EEG retrieval. https://arxiv.org/abs/2609.02582
- EEG2Video / SEED-DV (EEG while watching videos). https://github.com/XuanhaoLiu/EEG2Video
- *What EEG-to-Video Decoding Actually Recovers*: EEG carries coarse, semantic information. https://arxiv.org/abs/2505.21385
- EEGNet. https://arxiv.org/abs/1611.08024
- Banded ridge regression. https://pmc.ncbi.nlm.nih.gov/articles/PMC9807218/
- Net2Brain. https://github.com/cvai-roig-lab/Net2Brain
