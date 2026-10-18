# Track: time-staged experts

An alternative to the "proven" solution (CLIP-style matching). The idea comes from a study on this same dataset: the brain processes a 3-second video in stages, and a different kind of AI model matches each stage best. So we split the **EEG** into three time windows and use several "experts" (feature types) in each, then combine their scores.

**Current honest result (mean over sub-01..04): 9.1% top-1, 25.7% top-5, median rank 15.2.** Chance is 1.0%, 4.9% and 51.5. Everything runs on a laptop CPU with linear models; nothing is fine-tuned.

---

## 1. Idea

The windows cut the **brain signal**, not the video:

| Window | EEG time after video onset | What the brain does (from the literature) |
|---|---|---|
| early | 0.05–0.25 s | first look: brightness, contrast, edges, motion |
| middle | 0.25–0.80 s | recognizing objects and scenes |
| late | 0.80–3.00 s | following the action over time |

Source: "The Human Brain as a Dynamic Mixture of Expert Models in Video Understanding" (ICLR 2026, on the EEG Moments Dataset, which this hackathon's data comes from). It found low-level image features match the EEG best early, object models in the middle, and video models late. See `../RESEARCH_ALTERNATIVES.md`.

For every **expert × feature set × window**:
1. **Encoding model:** ridge regression from video features to the EEG of that window, trained on the subject's own training recordings.
2. **Predict** the EEG for each of the 102 candidate videos.
3. **Score:** correlate each test recording with each prediction, giving a 102 × 102 score matrix.

We don't hard-code which expert belongs to which window. Every expert is tried in every window, and the data decides the weights.

## 2. Experts

| Expert | Features | Model (exact version) |
|---|---|---|
| `lowlevel` | brightness, contrast, RGB, saturation, edges, frame-to-frame motion, plus brightness and motion on a 4×4 screen grid; 40 numbers per 0.1 s bin, 30 bins | none (hand-made, OpenCV) |
| `dino` | 8 frames (0.19–2.81 s); class token + mean patch token at blocks 4, 8 and 12; mean of the first 2 frames or of all 8 | DINOv2 ViT-S/14, timm `vit_small_patch14_dinov2.lvd142m`, timm 1.0.30 |
| `motion` | 3 clips of 16 frames (whole 3 s, first half, second half); class token + mean token at blocks 3, 9, 14 and 16 | MViT-v2-S, torchvision `mvit_v2_s`, weights `MViT_V2_S_Weights.KINETICS400_V1`, torchvision 0.29.1 |

**No AlexNet** and no CLIP. All models are public and pretrained; we only read their features.

## 3. Results (mean over sub-01..04)

Per expert, with its own three windows weighted equally (best feature set):

| Expert | top-1 | top-5 | median rank |
|---|---|---|---|
| lowlevel | 2.7% | 11.3% | — |
| dino (block 8, all frames) | 5.1% | 21.8% | 19.8 |
| motion (block 14, whole clip) | 5.6% | 21.1% | 20.1 |

Combined (`05_combine.py`), all three experts:

| Strategy | in-sample top-1 / top-5 / median | honest (LOSO) top-1 / top-5 / median |
|---|---|---|
| equal weights | 5.6% / 23.3% / 16.5 | 5.6% / 23.3% / 16.5 |
| per-window weights (early 2, middle 2, late 1) | 8.6% / 25.2% / 15.2 | 8.6% / 25.2% / 15.2 |
| **expert × window (used)** | 9.6% / 27.5% / 15.6 | **9.1% / 25.7% / 15.2** |
| greedy selection | 9.8% / 26.2% / 14.4 | 8.1% / 27.5% / 15.5 |

Notes on reading these numbers:
- Each subject has only 102 test recordings, so top-1 moves by about ±1% from luck alone. Top-5 and the median rank are more stable.
- The used strategy is the one with the best honest score, so its honest number is slightly optimistic.

## 4. Findings (for the presentation)

1. **Simple visuals only work early** (early 11.5% top-5, middle 8.6%, late 7.4%). That reproduces the first stage of the paper on our 17 channels.
2. **Object features (DINOv2) lift the middle window** from 8.6% to up to 19% top-5. Deeper layers help in the middle window but not early, which matches "objects are recognized after the first look."
3. **Intermediate layers work best overall** (DINOv2 block 8, MViT block 14). Other EEG studies report the same.
4. **The late window stays weak, even with the video model** (5–8% top-5). Possible reasons: our 17 electrodes may not cover the back of the head, where the paper found the late effect; the late signal is weak and the window is long; and every training recording is a single viewing.
5. **Most of the information sits in the first ~0.8 s.** The fitted weights are early 2, middle 2, late 1. This matches what the organizers said about the first ~0.5 s.
6. **Picking many settings on 4 subjects overfits.** Greedy selection dropped from 8.3% to 3.9% top-1 on an unseen subject before the motion expert existed. Fitting a few weights (one per window, or one per expert × window) doesn't. That's the reason for the honest LOSO protocol.

## 5. Validation protocol

- Dev subjects sub-01..04: their test labels are used **only** to score and to choose combination weights (allowed as validation and model selection).
- **Honest score (LOSO, leave one subject out):** choose the weights on 3 dev subjects, score on the 4th, and repeat for each. This estimates what sub-05 and sub-06 will get.
- Nothing is trained on test EEG. Test EEG is only z-scored with the subject's **training** mean and std.
- Each subject's encoding models use only that subject's training data.
- `tsx/submission.py` writes the files with rows in the original order, values 1001..1102, shape (102, 102), integers. It reloads and re-checks each file after saving.

## 6. Setup

This track needs its **own environment**. Installing `net2brain` (used by the organizers' notebooks) downgrades `timm` to 0.4.12, which can't load DINOv2.

~~~bash
cd ~/Projects/krabiki
python3.11 -m venv .venv-tsx
source .venv-tsx/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn opencv-python tqdm timm
~~~

The data folder `eeg-mind-reading-challenge/` must sit next to `track-timestaged/`, or `EEG_DATA_ROOT` must point to it.

## 7. Running

From `track-timestaged/`, with `.venv-tsx` active. Times were measured on a ThinkPad T14s, CPU only.

| Step | Command | Time |
|---|---|---|
| check data and submission format | `python 00_check_setup.py` | seconds |
| low-level features | `python 01_extract_lowlevel.py` | ~1.5 min |
| low-level expert | `python 02_lowlevel_expert.py` | ~1 min |
| DINOv2 features (downloads 88 MB) | `python 03_extract_dino.py` | ~10 min |
| object expert | `python 04_object_expert.py` | ~2 min |
| MViT-v2 features (downloads 132 MB, resumes after interruption) | `python 06_extract_motion.py` | ~1 h |
| motion expert | `python 07_motion_expert.py` | ~3 min |
| combine everything | `python 05_combine.py` | seconds |

Outputs (not in Git):
- `cache/*.npz`: video features
- `outputs/sims/<expert>_sub-XX.npz`: score matrices for every feature set and window, and every subject
- `outputs/submissions/<expert>/` and `outputs/submissions/ensemble/`: submission files
- `outputs/results.csv`: every result ever computed
- `outputs/ensemble.json`: the chosen combination

## 8. Combining with another approach

Score fusion: each approach scores the same 102 × 102 grid, scores are put on the same scale, and a weighted sum is ranked. In the Algonauts 2025 competition and in EEG retrieval papers (e.g. CORTIVA), combining approaches this way was what made the best entries win.

**What the other approach must save:** for every subject 1..6, one `.npy` file with a (102, 102) float array.
- **Rows** = test recordings in the **original order** of `tst/sub-XX_tst.npy`. Never sorted by label. (The organizers' example notebook sorts the rows with `np.argsort(gt)`; that must not happen here.)
- **Columns** = candidate videos 1001, 1002, …, 1102, in that order.
- **Values:** higher = more likely. Any scale works; the combiner z-scores each row.
- It's best to save the raw similarity *before* ranking (correlations, cosine similarities, or logits), not ranks.
- Files are needed for **all 6 subjects**, including sub-05 and sub-06, made with exactly the same pipeline.

**Import and combine:**
~~~bash
python 08_import_scores.py --name proven --pattern "/path/to/their/sims/sub-{s:02d}.npy"
python 05_combine.py
~~~
`08_import_scores.py` checks shape and values, then prints top-1/top-5 for sub-01..04. **These must match the numbers the other approach reports.** If they don't, rows or columns are in the wrong order. `05_combine.py` then treats the imported scores as one more expert (window `all`) and picks weights with the same honest protocol. To compare without our experts: `python 05_combine.py --experts proven`.

**Two things to check on the other side:**
1. **Is their number honest?** If their settings (epochs, learning rate, layers, windows) were chosen by looking at the sub-01..04 test results, their 14% is optimistic too. The fairest comparison uses the same LOSO idea: choose settings on 3 dev subjects, test on the 4th.
2. **Rules:** sub-01..04 test recordings must not be used as templates for sub-05/06, and no model may be trained on any test data.

**Why combining might help:** our experts are DINOv2 + MViT + hand-made features with linear encoding models, scored per time window. A CLIP-based contrastive model looks at the problem differently. If the two get *different* videos wrong, the sum beats both. If the combiner gives our experts weight 0, they don't add anything.

## 9. Code layout

~~~
track-timestaged/
├── 00_check_setup.py          data + submission format check
├── 01_extract_lowlevel.py     hand-made features      -> cache/lowlevel.npz
├── 02_lowlevel_expert.py      low-level expert         -> outputs/sims/lowlevel_*
├── 03_extract_dino.py         DINOv2 features          -> cache/dino.npz
├── 04_object_expert.py        object expert            -> outputs/sims/dino_*
├── 05_combine.py              combine all experts      -> outputs/submissions/ensemble/
├── 06_extract_motion.py       MViT-v2 features         -> cache/motion.npz
├── 07_motion_expert.py        motion expert            -> outputs/sims/motion_*
├── 08_import_scores.py        import another approach  -> outputs/sims/<name>_*
└── tsx/
    ├── data.py                loading, downsampling (500 -> 100 Hz)
    ├── video_features.py      hand-made features
    ├── dino.py                DINOv2 feature extraction
    ├── motion.py              MViT-v2 feature extraction
    ├── encoding.py            windows, ridge encoder, correlation scores, z-scoring
    ├── experts.py             runs an expert over all feature sets and windows
    ├── evaluate.py            ranking, top-1 / top-5 / median rank
    └── submission.py          writes and validates the submission files
~~~

## 10. Limitations and next ideas

- **Each subject's model only learns from ~950 single, noisy recordings.** Training one shared model on all 6 subjects (allowed) would give about 6× more data per video. This is probably the biggest remaining gain.
- **The late window is mostly unused.** Shorter late windows (0.8–1.5 s and 1.5–3 s) or coarser time steps might recover some signal.
- **The window boundaries come from a 128-channel study** and were not tuned on our data.
- **The video's sound is not used yet** (an audio model such as CLAP).
- **One-to-one matching:** each test video appears once per subject, so the final ranking could be forced to be one-to-one. This usually raises top-1 a lot, but it uses the structure of the test set. **Ask the organizers before using it.**
