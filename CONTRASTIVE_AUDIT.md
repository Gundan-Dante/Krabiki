# Linear contrastive head (InfoNCE on frozen features): perspective audit

Written by session krabiki-2f before running anything (2026-10-18). The experiment and its result are recorded as
M11 in `MODEL_BACKLOG.md`.

## What changes compared with ridge decoding (M8a)

**Ridge (current best)** fits a linear map W from the EEG window (935 values, noise-whitened, standardized) to the
standardized video vector by minimizing the **squared error on every feature dimension independently**, plus
`alpha·‖W‖²`. It's closed-form and convex, alpha is chosen by train-CV, and there's no optimizer randomness.
Retrieval then uses cosine similarity, which ridge never saw during fitting.

**InfoNCE** fits the same kind of map, but the loss is a **softmax over the candidates in the batch**. For each EEG
trial, the true video's cosine must beat the other N−1 videos' cosines (and symmetrically for each video). Concretely:

- It **optimizes ranking directly** (a smooth surrogate of top-1 in the batch) with the same cosine used at test time.
  Ridge optimizes reconstruction and gets ranking for free only if errors are isotropic.
- It **uses negatives**. Directions that reconstruct variance shared by all videos (useless for ranking) earn
  nothing, while directions that separate similar videos are rewarded. This is roughly what the reliability weighting
  (M8b) tried by hand.
- It ignores the norm of the prediction (cosine) and has one extra parameter, the temperature τ.
- It's **non-convex** (because of the normalization) and trained by SGD, so results depend on init, learning rate,
  early stopping and seed.

## What to expect with about 850–970 trials per subject and no pooling

- **Parameter count vs data:** W has 935 × 512 ≈ 480k parameters (935 × 768 for DINOv2), estimated from about 800
  trials per CV fold. Ridge copes because shrinkage gives the minimum-norm solution in the span of the data. InfoNCE
  has no such closed form and has to be held back with weight decay and early stopping.
- **Negatives:** a full batch gives about 650–800 negatives per positive, which is plenty in count. But each positive
  is a single noisy trial, with no repetitions to average.
- **The comparison case is the literature.** NICE (Song et al., 2023) and ATM (Li et al., NeurIPS 2024) train
  contrastive EEG–image models on THINGS-EEG2: about 16.5k training images per subject, each shown 4 times and
  averaged, with ~200-way test sets of trials averaged over ~80 repetitions. They use **nonlinear** spatio-temporal
  encoders. Reported 200-way top-1 is roughly 15% for NICE and roughly 28% for ATM (approximate figures from memory;
  check before quoting). Their gains come from **data volume, trial averaging and nonlinear encoders**, none of which
  we have. Our setup (single 3 s trials, about 950 per subject) is closer to their low-data ablations, where linear
  or ridge baselines are competitive.
- **Linear InfoNCE vs ridge.** With a linear map and frozen targets, ridge + cosine is already a strong, well-regularized
  surrogate for the same objective. The extra freedom InfoNCE has (re-weighting directions by how discriminative they
  are) is exactly what tends to overfit when trials are few.

## Risks

1. **Overfitting.** Training InfoNCE goes to about 0 quickly when parameters far exceed trials. Mitigation: weight
   decay, early stopping on an inner 15% split of training videos, warm start from ridge (variant C) with decay toward
   the ridge weights.
2. **Temperature collapse.** A learnable τ shrinks to sharpen a memorized training batch, and gradients get spiky.
   Mitigation: init 0.07, clamp to [0.01, 1], and early stopping on held-out retrieval rather than loss.
3. **SGD variance.** Unlike ridge, results move with the seed. Report 3 seeds.
4. **Selection noise.** A grid of about 36 configs × 6 subjects, chosen by train-CV, is a multiple-comparison setting.
   The M8b lesson (CV preferred a variant that dev didn't confirm) applies, so confirm with the paired dev test.
5. **Low ensemble diversity.** Same inputs, same targets, linear map: errors will correlate with M8a, so an ensemble
   gain is unlikely unless the contrastive map is genuinely different (for example, the two-tower variant).

## Honest prior

- **Most likely (about 50%): equivalent to M8a** within noise. Warm start C stays near the ridge solution, and A
  converges to a similar map.
- **About 25%: a small, significant gain** (+1–3 pts top-5), most plausibly from C, where InfoNCE fine-tunes ridge
  toward discriminative directions.
- **About 25%: worse.** A and B from scratch overfit, or the temperature dynamics hurt.
- **Ensemble value: low.** The residual correlation with M8a is expected to be high.

A negative result is still informative for the report: it shows that, at this data scale, the closed-form ridge
decoder is the right tool, and that contrastive learning's advantages in the literature come from data and
nonlinearity, not from the loss alone.

## Test plan (as assigned)

Inputs are identical to M8a: EEG 0–1.5 s, decimated, window 5:60, per-timepoint noise-whitened (encoding OOF
residuals), standardized with training stats. Video side: frozen CLIP B-32 f8 (then DINOv2-B patch), standardized with
training-video stats. Variants: **A** linear W (935 → D) + cosine + symmetric InfoNCE with learnable τ; **B** two towers
(935 → d, D → d), d ∈ {64, 128, 256}; **C** A initialized at the M8a ridge solution, with either decay toward the ridge
weights or plain weight decay. Training uses full-batch AdamW, a small lr × weight-decay grid, and early stopping on an
inner 15% split of the training videos.

Selection uses the shared outer 5-fold train-CV (KFold seed 0), scored by mean percentile rank plus expected top-1 and
top-5 of 102 (hypergeometric). Whiteners are re-estimated inside each outer fold. Dev is used only for confirmation:
a paired Wilcoxon test vs M8a, 3 seeds for the best variant, and an equal-weight ensemble diagnostic with M8a. The
timebox is about 1.5 h.

## Outcome (filled in after the runs; details are in MODEL_BACKLOG M12)

The prior mostly held. **As a replacement it's neutral to worse:** the warm start converges back to the ridge solution
(early stopping picks 0 steps), linear-from-scratch is worse by train-CV on 6/6 subjects but not significantly
different on dev (15.6 ± 1.5 / 45.0 ± 1.1 vs 14.7 / 42.2), and two towers are worse. **The surprise is ensemble
value**, which I rated "low". Weak contrastive heads err differently from ridge, and the CV-weighted blend
(0.75 ridge + 0.25 contrastive, w chosen by train-CV) improves on M8a on every seed. CLIP two-tower reaches
15.9 ± 1.1 / 45.5 ± 0.5 (paired p ≤ 2e-6), and the effect repeats with DINOv2. The CV gain is small relative to the
dev effect, so final weights should come from the shared N-member CV tool.
