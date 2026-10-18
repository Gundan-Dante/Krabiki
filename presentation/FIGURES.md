# Presentation figures

Regenerate everything on CPU with `PYTHONPATH=src python presentation/make_figures.py` (one function per figure; e.g. `fig1 fig5`).
Output: `presentation/figures/*.png` (200 dpi). Inputs are small copies of saved matrices and tables in `presentation/data/`; numbers come
from `MODEL_BACKLOG.md` / `PROGRESS.md` (M-ids in the labels) or are recomputed from the saved similarity matrices. "Dev" = test
recordings of subjects 1–4 (4 × 102 = 408 trials, 102 candidate videos each); chance is 1.0% top-1 and 4.9% top-5.
Removed from the earlier set (still in git history): the whitening-bug figure and the feature bar chart.

## 1. `fig1_progress.png`: how the score grew
Each bar adds one idea. We start at 2% exact matches with a standard image network and reach 32% with the current model (v3); the
true video is among the 5 best guesses 67% of the time (chance: 5%). Whiskers show how uncertain each number is with only 408 test
recordings. The last slot (v4, subject pooling) is not finished yet.

## 2. `fig2_encoding_vs_decoding.png`: two directions, one winner
Encoding predicts the brain signal for a video and compares it with the real recording. Decoding turns the recording into a description
of the scene and compares that with each video. Comparing descriptions works much better (14% vs 5%) because raw EEG is mostly noise.

## 3. `fig3_time_course.png`: when is the information in the EEG?
Top: the average EEG over all training videos (thin line per channel; thick line = overall amplitude). Bottom: how well a simple decoder
identifies the video from a 100 ms slice (0 = guessing). Information peaks 150–250 ms after the video starts and fades after about
600 ms, so we use 50–600 ms (shaded). Later slices are still above chance, so this is a choice that works best, not a hard cutoff.

## 4. `fig4_ensemble_anatomy.png`: diverse views, each earns its place
The final model blends seven "members", each looking at the video differently (image, video, trained network, mid-level image, hand-made
statistics, sound, low-level sound and image). (a) Weights chosen by cross-validation per subject: the two scored subjects lean on the
low-level members. (b) Members that look at different things give different scores (low overlap), which is why blending helps.
(c) Removing any one member makes the cross-validation score worse, most for the mid-level image and hand-made members; the
low-level sound+image member was not measured (that run was cut off).

## 5. `fig5_expected_score.png`: what to expect on the two scored subjects
Bars show the real test scores for subjects 1–4. Diamonds are the training-data estimate (cross-validation), available for all six.
For the scored subjects 5 and 6 (no labels) the hatched bars are projections: the cross-validation estimate scaled by the test/cross-validation
ratio seen on subjects 1–4 (whisker: lowest to highest ratio): about 26% (21–32) and 18% (14–22) top-1, 58% and 48% top-5. These are
estimates, not measurements. Subjects 5 and 6 are the two weakest by cross-validation, and our dev numbers were selected among many
variants, so the true values are likely at the lower end. An earlier audit applied the same logic to the previous model and projected about 15% / 40%
on average (AUDIT.md §5).

## 6. `fig6_subject_pooling.png`: borrowing a little data from other subjects
Adding the other subjects' recordings with a small weight raises the cross-validation score for every subject, most for the weakest subject
(sub-06, +14%). On the test set the effect for one decoder was small and not significant (12.7 / 41.7 → 13.0 / 41.7). This is still being
integrated (v4).

## 7. `fig7_recall_at_k.png`: where does the true video land?
For each recording we rank all 102 videos. The curve shows how often the true one is within the first k guesses: 32% at k=1, 67% at
k=5, against a dashed chance line. Every modelling step lifts the whole curve.

## 8. `fig8_examples.png`: one hit and two near misses
Subject 1, the model's three best guesses next to the true video (middle frame). The first recording is a correct first guess. In the others
the true video is ranked 2nd and 4th, and the wrong guesses are partly similar scenes (water, outdoors). Examples were picked to illustrate
(one hit, two near misses), not at random.

## 9. `fig9_diagnostics.png`: two diagnostics
Left: a small neural network trails our simple decoder, but the gap shrinks as more training data is used (0.081 → 0.071 → 0.061), so the
data, not the idea, is the limit. Right: videos with more motion are slightly harder to identify (0.201 vs 0.179 for static ones; lower
is better), the opposite of what we first expected from a video model.
