# EEGNet (Stage 3): what we tried, what we hoped for, and why it fell short

Short write-up of the trained EEG encoder experiment (MODEL_BACKLOG.md **M14**, krabiki-2f; details in
`CONTRASTIVE_AUDIT.md`). Code: `src/eegmr/stage3.py`, `scripts/train_stage3.py`, and the `s3:` member type in
`scripts/ensemble.py` (track-matching). Numbers: train-CV = mean percentile rank of the true video on held-out training
folds (lower is better); dev = sub-01..04 test sets, top-1 / top-5 (chance 1.0 / 4.9).

## 1. What EEGNet is
EEGNet [1] is a compact convolutional network built for EEG. Its three blocks mirror classic EEG analysis:
1. **Temporal convolution:** learned frequency filters (which rhythms matter).
2. **Depthwise spatial convolution:** for each temporal filter, a learned weighting of the electrodes (where on the scalp).
3. **Separable convolution + pooling:** combines these spatio-temporal patterns into a compact feature vector.

It has only thousands to tens of thousands of parameters, designed for small EEG datasets. We used the braindecode
implementation, and compared it with ShallowFBCSPNet [2] (a band-power network) and a small MLP.

## 2. What we hoped for
Every other member of our ensemble uses the **same EEG side**: a linear ridge decoder on noise-whitened EEG (50–600 ms).
Their diversity comes only from the video side (DINOv2, V-JEPA, AlexNet, CLAP, …). A trained EEG encoder was the attempt
to also improve the **EEG side**:
- capture **nonlinear spatio-temporal patterns** that a linear map can't express;
- **learn its own filters** instead of using a fixed window;
- train with a **contrastive objective** (symmetric InfoNCE [3], as in CLIP [4]) that directly optimizes ranking the right
  video first.

This is the approach of the strongest EEG-to-image retrieval work: NICE [5] and ATM [6] train EEG encoders contrastively
against CLIP features on THINGS-EEG2 [7].

## 3. What we did
- **Inputs:** the same noise-whitened, standardized EEG as our decoders (per-timepoint Ledoit-Wolf whitening estimated on
  training data only). EEGNet used a 0–800 ms window. **Targets:** frozen CLIP B-32 features (and DINOv2-patch).
- **Ladder** (cheapest first): MLP head → EEGNet / ShallowFBCSPNet → + augmentation (±20 ms shift, channel dropout, noise,
  mixup) → (ATM-style block, skipped since rung 2 was far from ridge).
- **Training:** per subject (subject pooling was paused), AdamW, symmetric InfoNCE with learnable temperature, early
  stopping on an inner 15% split of the training videos, 3 seeds. Selection by train-CV (KFold seed 0); dev only to confirm.

## 4. Results

| Model | Train-CV (lower = better) | Dev top-1 / top-5 |
|---|---|---|
| Ridge decoder (M8a, CLIP, whitened) | **0.232** | **14.7 / 42.2** |
| EEGNet alone | 0.293 | 10.0 ± 0.4 / 30.2 ± 1.6 (worse, p ≤ 8e-7) |
| MLP head alone | ≈ linear (0.243) | 15.7 ± 1.1 / 44.6 ± 0.9 (CV said worse; n.s.) |
| ShallowFBCSPNet | ≈ chance (0.41–0.45) | – |
| **0.75 × ridge + 0.25 × EEGNet** | **0.224 (better on 6/6 subjects)** | **15.5 ± 0.2 / 45.2 ± 1.4 (p 3e-8 … 5e-4)** |
| Inside the full ensemble, with CSLS | +0.0004 LOFO | not adopted (complexity tie-break) |
| Inside the full ensemble, without CSLS | +0.0037 LOFO | kept in the no-CSLS backup `final_nocsls` |

- A window control (linear / MLP on the same 0–800 ms window: 0.2305 / 0.2291) shows that EEGNet's ensemble gain comes from
  the convolutional architecture, not from the longer window.
- Augmentation slightly hurt.

**Learning curves** (train-CV at 50 / 75 / 100% of each subject's training trials):

| | 50% | 75% | 100% |
|---|---|---|---|
| Ridge | 0.275 | 0.249 | 0.232 |
| EEGNet | 0.356 | 0.320 | 0.292 |
| Gap (EEGNet − ridge) | 0.081 | 0.071 | 0.061 |
| EEGNet's gain as an ensemble member | −0.001 | +0.0045 | +0.008 |

The gap shrinks by ~0.02 per doubling of data. Extrapolated, EEGNet needs **~8× more data** to match ridge alone.

## 5. Why it fell short
1. **Too little data (the main reason).** ~850–970 *single, noisy* trials per subject. The work that makes trained EEG
   encoders shine uses far more data, e.g. THINGS-EEG2 with 16,540 training images × 4 repetitions × 10 subjects, and
   test trials averaged over 80 repetitions [7]. The learning curves above show we're firmly in the data-limited regime.
2. **Fast overfitting.** Early stopping halted training after a few dozen steps, before much real signal was learned.
   Augmentation didn't fix it.
3. **Little nonlinearity to exploit at this signal-to-noise ratio.** The MLP did no better than linear. With
   single-trial EEG this noisy, a well-regularized linear model with good noise whitening is close to the best available.
4. **The ensemble already covers its contribution.** As a helper next to one ridge decoder it adds signal. In the final
   ensemble (CSLS + diverse low-level, temporal and audio members), that contribution is mostly absorbed (+0.0004).

## 6. Status and outlook
**Parked, not dropped.** EEGNet is viable as an ensemble member and is data-limited, not a dead end. It's the component
that would benefit most from **subject pooling** (~6× more training data). Our single-member pooling test (M26) already
helps the linear decoders, the weakest subject most. A pooled EEGNet is the natural next experiment.

## References
1. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact
   convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
   arXiv:1611.08024.
2. Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization.
   *Human Brain Mapping*, 38(11), 5391–5420. (ShallowFBCSPNet; braindecode.)
3. van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748.
   (InfoNCE.)
4. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision (CLIP). arXiv:2103.00020.
5. Song, Y., et al. (2024). Decoding natural images from EEG for object recognition (NICE). ICLR 2024. arXiv:2308.13234.
6. Li, D., et al. (2024). Visual decoding and reconstruction via EEG embeddings with guided diffusion (ATM). NeurIPS 2024.
   arXiv:2403.07721.
7. Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual
   object recognition (THINGS-EEG2). *NeuroImage*, 264, 119754.
