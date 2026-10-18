# Presentation figures

Regenerate everything with `PYTHONPATH=src python presentation/make_figures.py` (one function per figure; pass `fig3 fig4` to
make only some). Output: `presentation/figures/*.png` (200 dpi). Inputs are in `presentation/data/` (copies of the saved
similarity matrices and CV tables from the model tracks). Every number comes from `MODEL_BACKLOG.md` or is recomputed from
those matrices. "Dev" = test recordings of subjects 1–4 (4 × 102 = 408 trials, 102 candidate videos each); chance is 1.0% top-1
and 4.9% top-5.

## 1. `fig1_progress.png`: how the score grew
Each bar adds one idea to the previous model. The baseline, which predicts the brain signal from simple image features, finds the
exact video 2% of the time. Our final model (M11), which reads the brain signal and compares it to two kinds of video features,
gets 15%, and 45% of the time the true video is among its 5 best guesses (chance: 5%). The whiskers show how uncertain
each number is with only 408 test recordings, so the last small steps (14.0 → 14.7 → 15.2%) are not individually
convincing, while the jumps from baseline to CLIP to decoding clearly are.

## 2. `fig2_encoding_vs_decoding.png`: two directions, one winner
Encoding predicts what the brain should look like for a video and checks it against the real recording. Decoding does it the
other way round: it turns the recording into a description of the scene and checks that against each video. Comparing in
the "description" space works much better (14% vs 5% exact matches) because the raw EEG is mostly noise, while the
description keeps only the gist of the scene.

## 3. `fig3_whitening_bug.png`: a bug that disabled the regularization knob
Left: a ridge regression shrinks weak signal directions more than strong ones, but our first pipeline had a rescaling step
that made every direction equally strong (flat line), so the shrinkage could not tell signal from noise. Removing that step
(blue) restores a natural spread. Right: with the step removed, accuracy rises (top-1 4.2 → 5.4%, top-5 16.9 → 21.1%; paired
test p = 2e-13). Computed here from subject 1's CLIP features; the accuracy bars are the logged results of the two pipelines.

## 4. `fig4_time_course.png`: when is the information in the EEG?
Top: the average EEG over all training videos, one thin line per channel and a black line for overall amplitude (RMS over channels), with the 50–600 ms window we match on shaded. Bottom:
how well a simple decoder can identify the video from just a 100 ms slice of EEG (0 = guessing, higher = better; measured by
cross-validation on training data only). Information peaks about 150–250 ms after the video starts and then fades, which is
why we only compare the first 600 ms. Later slices are still better than chance, so the shaded window is a choice that works
best, not a hard cutoff. (Skill is plotted as 0.5 minus the average relative rank of the true video.)

## 5. `fig5_per_subject.png`: every subject
Left: real test accuracy for the four subjects whose answers we have (top-1 13–21%, top-5 41–49%, all far above chance).
Right: a check that needs no answers, so it also covers subjects 5 and 6. The bars are the average position of the true
video when predicting held-out training videos (0 = always first, 0.5 = guessing). Diamonds mark the same quantity on the
real test set for subjects 1–4. The training check is deliberately pessimistic (it trains on only 80% of the data), so the
test scores are better than the bars. Subjects 5 and 6 are in the same range as the labelled ones (sub-06 is the weakest), so
we expect them to work, probably a bit below subjects 1–4.

## 6. `fig6_recall_at_k.png`: where does the true video land?
For each recording we rank all 102 videos. The curve shows how often the true one is within the first k guesses: 15% at
k=1, 45% at k=5, 59% at k=10, 78% at k=20, against a dashed straight chance line. Even when the first guess is wrong, the
right video is usually near the top, and every modelling step lifts the whole curve, not just one point.

## 7. `fig7_examples.png`: one hit and two near misses
Subject 1, three recordings, the model's three best guesses next to the true video (middle frame). The first is a correct
first guess. In the other two the true video is ranked 2nd and 4th. The wrong guesses are only sometimes similar scenes:
for the outdoor scene in row 3 they are water and outdoor footage, which is plausibly gist-level, but for the sawing video in
row 1 they are babies eating, which is not similar. So the examples are suggestive, not proof that the EEG carries scene gist.
The three examples were picked to illustrate (one hit, two near misses), not at random.

## 8. `fig8_features.png`: which video features?
Top-5 accuracy by the kind of video description. Left (encoding): a generic image network (AlexNet) is far behind CLIP,
which was trained on images paired with text. Right (decoding): CLIP and DINOv2 patch features perform about equally (42%),
DINOv2's whole-image summary is weaker (34%), and using both CLIP and DINOv2 together is best (45%). The gain from combining
(+3 points) is within the uncertainty of 408 recordings, but a paired test per recording favours it (p = 0.002).
AlexNet was only run with the encoding head, so the two groups are not directly comparable.
