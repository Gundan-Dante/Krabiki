# Presentation guide: Krabiki, EEG mind reading

A slide-by-slide script with every detail needed to present the work: what to show, what to say, the exact numbers, and
answers to likely questions. Figures live on branch `track-viz` in `presentation/figures/` (regenerate with
`CUDA_VISIBLE_DEVICES= PYTHONPATH=src python presentation/make_figures.py`); captions are in `presentation/FIGURES.md`.
Background docs: `IMPROVEMENTS.md` (each step), `ARCHITECTURE.md` (pipeline), `AUDIT.md` (checks), `EEGNET.md`,
`MODEL_BACKLOG.md` (every experiment, M0–M28).

**Numbers convention.** "Dev" = the test sets of subjects 1–4 (labels available; 4 × 102 recordings). The scored subjects are
5 and 6, whose test labels are hidden. Chance: **1.0% top-1, 4.9% top-5, median rank 51.5**. Every choice was made on
**training data only** (5-fold cross-validation over each subject's training videos); dev only confirmed it.

---

## Slide 1: The task (30 s)
- 6 people watched **1,102 three-second video clips with sound** while EEG was recorded: 17 electrodes at the back of the head
  (Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2), 500 Hz.
- 1,000 clips for training. **102 new test clips that never appear in training**, so this is zero-shot retrieval.
- For each test recording: **rank all 102 candidate clips** by how likely they caused it. Scored on top-1 / top-5 for
  subjects 5 and 6.
- Say: *"We can't learn 'EEG → clip ID'. We have to map brain activity and videos into a space where they can be compared."*

## Slide 2: Headline result (30 s), fig1_progress.png
- **From 2.0% to 33.8% top-1 and from 8.6% to 68.1% top-5** on dev (chance 1% / 5%).
- Final model: `submissions/final_v4b_csls/` (v4b). Median rank of the true clip: **2.6 out of 102**.
- Say: *"In two out of three recordings, the true clip is among our top 5 guesses out of 102."*

## Slide 3: The core idea: decode, don't encode, fig2_encoding_vs_decoding.png
- **Organizers' baseline (encoding):** video features → predict the full EEG (17 × 300 numbers) → correlate with the measured EEG.
  Most of those 5,100 numbers are noise, so the comparison happens in a noisy space. Result 2.0% / 8.6%.
- **Ours (decoding):** EEG window **50–600 ms** (17 channels × 55 time points = 935 numbers) → **ridge regression** → a
  *predicted* feature vector of the video → compare with each candidate's real feature vector by **cosine similarity**.
- **Why cosine:** ridge shrinks predictions toward zero, so their length is meaningless; only the direction (which content)
  counts.
- **Effect:** with the same CLIP features, 5.4% → **14.0%** top-1, the biggest single jump.
- **Alpha** (the ridge penalty) is chosen by *retrieval cross-validation*: hold out 1/5 of the training clips, rank them, and keep the
  alpha that ranks best.

## Slide 4: Two "whitenings" (the bug and the fix)
- **The bug (removed):** the recipe rescaled features after PCA so that every direction was equally strong (ZᵀZ = n·I). Ridge
  could then only shrink all predictions uniformly, and the matching ignores scale, so **alpha had no effect at all**.
  Fixing it: 4.2% → 5.4%. Lesson: *never standardize a ridge's inputs after PCA*.
- **The gain (adopted): EEG noise whitening.** EEG noise is shared across neighbouring electrodes. From training data we
  estimate, **for each time point, how the noise is shared across the 17 electrodes** (Ledoit-Wolf covariance of the
  residuals the model can't explain), and transform the EEG so that noise becomes uncorrelated. Decoder 14.0 / 36.0 →
  **14.7 / 42.2**; the old encoder nearly doubled (5.4 → 8.8).

## Slide 5: When is the information there? fig3_time_course.png
- Top: average evoked response. Bottom: how well 100 ms windows identify the clip (training CV).
- Information **peaks at ~150–250 ms** after onset and is mostly gone after ~600 ms, so we use **50–600 ms** (cross-validation chose
  it on all 6 subjects; wider windows were worse).
- Say: *"The decoder reads the brain's first look at the clip."*

## Slide 6: An ensemble of different views, fig4_ensemble_anatomy.png
Seven members share the same EEG pipeline but predict **different descriptions of the video**:

| Family | Member (model, version) | Role |
|---|---|---|
| Semantic image | DINOv2-B/14 patch tokens (timm `vit_base_patch14_dinov2.lvd142m`), **subject-pooled** | objects, shapes |
| Video dynamics | V-JEPA 2-L (`facebook/vjepa2-vitl-fpc64-256`, 16 frames) | motion, temporal structure |
| Ranking objective | contrastive two-tower head on CLIP B-32 (`laion2b_s34b_b79k`), InfoNCE, 3 seeds | trained to rank, not regress |
| Low-level spatial | AlexNet conv5 (torchvision IMAGENET1K_V1), 6×6 pooled, **subject-pooled** | edges, texture, layout |
| Hand-crafted | colour, spatial frequency, edges, motion / optical flow descriptors | interpretable low-level |
| Low-level over time | "lowav": per 100 ms, brightness, contrast, edges, motion + loudness, onsets, 8 mel bands, **subject-pooled** | timing of visual and sound changes |
| Audio | LAION-CLAP (`laion/clap-htsat-unfused`, pinned revision) | the sound of the clip |

- **(a) weights:** chosen by cross-validation. Final global weights: pooled DINOv2 .2, V-JEPA .1, contrastive .1, pooled AlexNet
  conv5 .2, descriptors .1, CLAP .1, pooled lowav .2.
- **(b) overlap:** the semantic trio overlaps most (DINOv2 ↔ V-JEPA 0.73); CLAP and lowav are the most unique (0.17–0.41).
- **(c) remove one:** every member costs score when dropped (conv5 −0.0104, descriptors −0.0106, DINOv2 −0.0073,
  contrastive −0.0046, V-JEPA −0.0032, CLAP −0.0013).
- Say: *"Different views make different mistakes. A bigger CLIP was worse; diversity beat size."*

## Slide 7: Why low-level features matter so much
- All 17 electrodes sit over **visual cortex**, which responds strongly to *how things look* (edges, contrast, texture,
  motion), not only to *what* they are.
- Adding AlexNet conv5, the descriptors, lowav and CLAP: 24.0 / 57.1 → **32.4 / 66.7**, the second-biggest jump.
- For the weakest subjects (5, 6), cross-validation gives lowav the **largest** weight.
- Consistent with the literature: early EEG matches low-level features, later EEG matches semantic ones
  (Azamat's research summary, the ICLR 2026 "dynamic mixture of experts" paper).

## Slide 8: Hub correction and the train/test shift
- **Hubs:** some candidate clips score high for *every* recording. **CSLS** lowers each candidate's score by how popular it is
  across all recordings. It's label-free, applied per member, k=5, and **approved by the organizers**. On the CLIP decoder alone: 14.7 / 42.2 →
  17.4 / 48.5.
- **Train/test shift:** test recordings are 22–37% less noisy than training trials (likely averaged over repeats) and their
  mean evoked response is *anti*-correlated with the training mean (−0.68 to −0.94). **Test-set centering** (subtract the
  test set's own mean, allowed as unsupervised normalization) removes that shared offset: a small, consistent gain.

## Slide 9: Learning from other people (subject pooling), fig6_subject_pooling.png
- Each subject has only ~900 noisy single trials, and the decoders are data-limited. All 6 people saw largely the same training clips.
- Each subject's decoder also learns from **the other subjects' training trials at a small weight λ** (5–20%), with every
  subject's EEG whitened with its own noise model. λ is chosen per subject by cross-validation.
- **Zero-shot evaluation:** in cross-validation, other subjects' recordings of the held-out clips are removed, as at test time.
- Cross-validation picked λ > 0 in **36 / 36** subject × member cases; **full pooling (λ = 1) hurts**, because brains differ.
- **The weakest subjects gain the most** (single member: sub-06 +14%).
- Final: pooling the 3 heaviest decoders (DINOv2, AlexNet conv5, lowav): CV 0.2774 → **0.2872**, dev **33.8 / 68.1**. Pooling all six
  added nothing more.

## Slide 10: Where does the true clip land? fig7_recall_at_k.png and examples fig8_examples.png
- Recall curves: v4b reaches **34% at rank 1 and 68% at rank 5**, far above the earlier models.
- Examples: one hit and two near misses. Be honest: wrong guesses are *sometimes* semantically similar (water scenes for
  outdoor nature), not always.

## Slide 11: What to expect on the scored subjects, fig5_expected_score.png
- Subjects 5 and 6 have no test labels, so we **estimate**: each subject's training cross-validation × the CV-to-test ratio seen on
  subjects 1–4 (×1.95 top-1, ×1.54 top-5).
- **sub-05: ~26% top-1 (21–29), ~60% top-5 (54–73). sub-06: ~18% (15–21), ~49% (44–60).** Expected average about **22% / 55%**.
- They're the two weakest subjects by cross-validation (sub-06 has the fewest trials: 850), so the estimate is **probably optimistic**.
- Label-free sanity: 62 / 54 distinct top-1 picks out of 102, and no clip picked more than 5 times, so there's no collapse.

## Slide 12: How we know it's real (integrity)
- **Training-only selection**; dev was used only to confirm (paired Wilcoxon per recording).
- **Shuffle controls:** with clip ↔ EEG pairs scrambled in training, the full pipeline falls to **chance** (final_csls: 1.0 / 4.7;
  v3: 1.0 / 3.9 and 1.0 / 5.9).
- **Code audit:** test EEG enters only for centering and CSLS (unsupervised); labels only for scoring; outputs don't depend on
  row order; reruns are bit-identical.
- **Independent reproduction:** key results were re-implemented independently and matched exactly. A teammate reproduced them on
  different hardware.
- **Excluded data:** we found that a public dataset (EEG Moments, OpenNeuro ds008257) likely contains this challenge's EEG,
  and **never used it**. The organizers were informed.
- **Reproducibility:** an inference bundle runs on a laptop CPU with numpy + scipy in seconds and reproduces the submission
  files bit-for-bit.

## Slide 13: What didn't work (shows rigor), fig9_diagnostics.png
| Tried | Outcome | Why |
|---|---|---|
| More frames (8 → 32) | no gain | frames of a 3 s clip are near-identical for CLIP |
| Bigger CLIP (ViT-L/14) | worse | EEG carries gist, not detail |
| Extra / wider windows | worse | the information sits in the first ~600 ms |
| THINGS-EEG2 transfer (same 17 channels) | ~2× chance only | 100 ms photos ≠ 3 s videos |
| EEGNet (trained EEG network) | weaker than ridge alone | data-limited; the gap shrinks ~0.02 per doubling of data |
| Motion gate | no gain | *every* model decodes moving clips worse |
| Pooling all six decoders | +0.0003 | the three heaviest members capture the gain |

## Slide 14: Takeaways
1. **Decoding into a feature space beats predicting EEG**, and noise whitening makes single-trial EEG usable.
2. **Diverse representations** (semantic, temporal, low-level, audio) each explain a different part of the brain signal.
   Diversity beats model size.
3. **Low-level vision matters** for posterior EEG, especially for the weakest subjects.
4. **A little shared data from other subjects helps**, most for those who need it.
5. **Everything was selected on training data and checked with controls.**

---

## Likely questions and answers
- **"Isn't a 7-member ensemble just stacking models?"** Each member is a different hypothesis about what visual cortex
  encodes, each one costs score when removed (drop-one test), and we dropped members that didn't earn their place (CLIP,
  CLIP block-8, AlexNet conv2, EEGNet).
- **"Did you tune on the test set?"** No. Weights, alphas, windows and λ were chosen by cross-validation on training clips.
  Dev labels (subjects 1–4) only confirmed. About 10 candidate decisions were screened on dev, so the dev numbers are somewhat
  optimistic; we say so.
- **"Is CSLS cheating?"** It uses no labels, only the similarity matrix, and the organizers approved it. We also kept a no-CSLS
  backup (`final_v3_nocsls`, 25.2 / 58.1).
- **"Why ridge and not deep learning?"** With ~900 noisy single trials per subject, a well-regularized linear decoder with
  noise whitening is near-optimal. EEGNet is data-limited (needs ~8× more data by its learning curve). Pooling is the path forward.
- **"Why are subjects 5 and 6 lower?"** They're the two weakest by cross-validation (sub-06 has the fewest trials). Pooling and the
  low-level / audio members help them most.
- **"How do you reproduce it?"** `submissions/README.md` has the commands. The inference bundle (`bundle/krabiki_final_v4b_bundle.zip`)
  reproduces the files on a CPU in seconds.

## Credits
- **Pipeline, decoding, whitening fix, audits, coordination:** krabiki sessions on `track-proven` / `track-matching`
  (Claude agents directed by Kanat).
- **Audio (CLAP) and low-level AV features (lowav), super-ensemble, cross-hardware reproduction:** Alikhan (`track-audio`).
- **Time-staged experts, motion gate, research survey:** Azamat (`main`).
