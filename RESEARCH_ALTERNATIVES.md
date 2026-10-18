# Alternative approaches: research summary

Purpose: ideas that are **different** from the "proven" track (CLIP features + ridge or contrastive matching), so the team can decide whether to try a second, independent solution.

All summaries are paraphrased (content rephrased for compliance with licensing restrictions). Links go to the sources.

---

## 1. The most important finding: we know where the data comes from

The hackathon data almost certainly comes from the **EEG Moments Dataset (EMD)**, published in August 2026 by Gifford and colleagues.
- 6 people watched **1,102 three-second videos with sound** while keeping their eyes on a central point.
- The original recording used 128 electrodes at a higher sampling rate. The hackathon gives us 17 electrodes at 500 Hz.
- The videos are the same ones used in the **BOLD Moments Dataset (BMD)**, an fMRI study of 10 other people (Lahner et al., *Nature Communications* 2024). There, the 1,000 training videos were shown 3 times and the 102 test videos 10 times. The test recordings we have are probably averages over repeats, which would explain why they are cleaner.
- BMD comes with **human annotations for every video**: at least 5 labels each for objects, scenes, and actions, text descriptions, a spoken transcription, and memorability scores.
- Sources: [EMD paper](https://arxiv.org/abs/2608.28768), [BMD paper](https://www.nature.com/articles/s41467-024-50310-3), [BMD code](https://github.com/blahner/BOLDMomentsDataset), [BMD data](https://github.com/OpenNeuroDatasets/ds005165)

**Correction:** earlier I guessed the data came from SEED-DV. That was wrong. EMD matches exactly: 6 people, 1,102 videos, a 1,000/102 split, 3 s clips with audio.

**Warnings:**
1. **Do not download or use EMD's original EEG for sub-05 or sub-06.** It almost certainly contains the hidden test answers. That is cheating, and it will be obvious, because the organizers are likely connected to the dataset authors.
2. The BMD **annotations** (labels, captions) and **fMRI** for the test videos are public and were released before the hackathon. Formally they describe the videos, not our subjects, but this is a gray area. **Ask the organizers** before using them.

The group behind EMD and BMD (Gifford, Cichy, Roig) also maintains **Net2Brain**, the toolbox the example notebooks use. So the judges probably know this literature well. Approaches grounded in it will land well.

---

## 2. Most relevant paper: how the brain processes these videos over time

**"The Human Brain as a Dynamic Mixture of Expert Models in Video Understanding" (ICLR 2026)**, on this same EEG dataset. [OpenReview](https://openreview.net/forum?id=bSsNSfyj8m), [summary](https://github.com/zhaoyang97/Paper-Notes-en/blob/main/docs/ICLR2026/computational_biology/the_human_brain_as_a_dynamic_mixture_of_expert_models_in_video_understanding.md). Code is part of Net2Brain.

- The authors compared **110 image and video AI models** with the EEG, moment by moment.
- Main finding: **no single AI model matches the brain the whole time.** The best match changes during the 3 seconds:

| Time after video start | What the brain is doing (back of head) | Best-matching AI model |
|---|---|---|
| 0.06–0.24 s | Simple visual features (edges, brightness) | AlexNet, early layers |
| 0.24–0.8 s | Recognizing objects | Image models, late layers (e.g. DenseNet) |
| 0.8–3 s | Following the action over time | Video models (e.g. MViT-v2) |

- Electrodes at the front of the head only carry information early (peak around 0.5 s, nothing after 0.8 s).
- Self-supervised models matched best in the object stage. Video models with memory over time (state-space models) helped in the middle stage.

**Why it matters:** the organizers' baseline only uses one model (AlexNet) and only 50–600 ms. This paper says we are throwing away the 0.8–3 s part, where video models (not image models) match best.

---

## 3. Alternative solutions (ranked by fit for us)

### Option A: "Time-staged experts" (most promising, novel, grounded in the paper above)
**Idea:** use a different AI model for each part of the 3-second response, then combine them.
- Early window (about 0.05–0.25 s): low-level features (AlexNet early layers, brightness, contrast, motion energy).
- Middle window (about 0.25–0.8 s): object features (DINOv2 or CLIP, intermediate and late layers).
- Late window (about 0.8–3 s): **video model** features (VideoMAE, MViT, or V-JEPA 2) plus audio.
- Fit a simple model per window and per feature set, score all 102 candidates per window, and combine the scores, with weights chosen on sub-01 to sub-04.

**Why it's good:** it directly uses published knowledge about this exact dataset, it's cheap (linear models), and it's easy to explain in the presentation: "the brain switches experts, so we switch experts too."

**Risk:** low. It can also be merged into the proven track as an ensemble member.

### Option B: "Match by relationships" (similarity-profile matching)
**Idea:** instead of comparing a brain signal with a video directly, compare **how each one relates to the 1,000 training videos**.
- For a test brain signal: compute how similar it is to each training brain signal of the same person. That gives a profile of 1,000 numbers.
- For a candidate video: compute how similar it is to each of the 1,000 training videos in AI-feature space. That gives another profile of 1,000 numbers.
- The right video should have a profile that looks like the brain signal's profile.

**Why it's good:** this is the logic behind representational similarity analysis (RSA), which is standard in this research group. Each person is only compared with their own training data, so differences between people matter much less. No neural network training is needed.

**Rules check:** it only uses training data as reference points, which is allowed. It does not use dev-subject test data as templates.

Related reading: [Structural Representations for Brain–Vision Matching](https://arxiv.org/html/2603.07077v1), and the RSA method in the ICLR paper above.

### Option C: Use words (captions and labels) as the target
**Idea:** EEG carries the gist ("dog running on a beach"). Turn each video's human description (from BMD) into a sentence embedding and match the brain signal to that, alone or together with image features.
- Several papers report that text-based targets help EEG decoding, because they capture meaning without pixel noise: [Bridging EEG, Vision, and Language](https://arxiv.org/html/2605.24523), [RealMind](https://arxiv.org/html/2410.23754), [Interpretable EEG-to-Image with Semantic Prompts](https://arxiv.org/abs/2507.07157).
- Instead of using BMD captions, we could generate captions ourselves with a public vision-language model. That is safer for the rules, but slower on CPU.

**Risk:** needs an organizer OK if we use BMD annotations.

### Option D: Copy the lesson from Algonauts 2025 (the biggest brain-prediction competition)
[Insights from the Algonauts 2025 Winners](https://arxiv.org/html/2508.10784), with the winner [TRIBE (Meta)](https://arxiv.org/html/2507.22229v1):
- **Nobody trained their own feature models.** Everyone used pretrained ones.
- **Combining vision + audio + language was essential.**
- **The architecture didn't really matter:** transformers, recurrent networks, and even purely linear models scored almost the same.
- **Ensembling decided the winner:** averaging many model variants, weighted by how well each did on validation.
- TRIBE also dropped modalities at random during training (for example, hiding the audio) to make the model robust.

Related: [CORTIVA](https://arxiv.org/html/2608.01355) shows for EEG/MEG retrieval that **combining the final scores** of several separate models beats merging all features into one embedding (73.5% top-1 on a 200-way image benchmark).

**Takeaway:** spend time on more feature types and good score combination, not on fancy architectures.

### Option E: Cleaner EEG before anything else (cheap boost for every approach)
- **Multivariate noise normalization (MVNN):** whiten the EEG using its noise covariance. It's standard in this research group's pipelines, including the ICLR paper. Evidence is mixed in some studies, so test it ([Guggenmos et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29499313), [critique](https://pubmed.ncbi.nlm.nih.gov/34728244/)).
- **Use intermediate layers and blurred images:** EEG sees coarse structure (low spatial frequency) better than fine detail, so intermediate layers of vision models beat the final layer ([Structural Representations paper](https://arxiv.org/html/2603.07077v1)).
- **Meta's MEG study:** a learned decoder beat linear decoders by 7×, and DINOv2 matched later brain responses best ([Benchetrit et al.](https://arxiv.org/html/2310.19812)).

### Option F: EEG "foundation models" (pretrained brain readers): lower priority
- Models such as **CBraMod** ([paper](https://arxiv.org/abs/2412.07236)) are pretrained on huge EEG collections and then fine-tuned.
- Problems for us: they are mostly tested on medical and brain-computer-interface tasks, not visual decoding. Our 17-channel layout is unknown. Fine-tuning is slow on CPU. A recent study warns that their gains often don't hold up under careful controls ([What Do EEG Foundation Models Capture?](https://arxiv.org/pdf/2605.11410v2), [negative controls](https://arxiv.org/html/2607.24519v2)).
- Verdict: interesting for novelty points, but a risky use of 2 days.

### Option G: Classic "covariance / Riemannian" EEG methods: low priority
- These are strong for aligning different people (re-centering each person's data) in motor-imagery tasks ([ITSA](https://arxiv.org/abs/2508.08216)).
- They mostly ignore the timing of the response, and timing is exactly what carries video information here. Possibly useful only as a person-alignment step.

### Option H: Cross-person alignment ("shared space")
- Methods such as the shared response model and hyperalignment map every person into a common space using responses to the same stimuli ([hyperalignment review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7266639/), [shared-space decoding](https://arxiv.org/html/2607.19394v1)).
- We have this: all 6 people watched many of the same training videos. We could learn a mapping of each person into a shared space, pool everyone's data, and decode there.
- Good fit for "train on all 6 people" and cheap with linear algebra.

### Option I: One-to-one assignment at the end (ask organizers)
Each test video was shown exactly once per person, so the final ranking can be forced to be one-to-one (Hungarian algorithm or Sinkhorn). This often raises top-1 a lot, but it exploits the test structure. **Ask before using it.**

---

## 4. Ideas not worth our time
- **Generating or reconstructing videos from EEG** (diffusion models, as in [EEG2Video](https://github.com/XuanhaoLiu/EEG2Video) and [DynaMind](https://arxiv.org/html/2509.01177v1)). They look impressive, but they need GPUs and don't improve ranking. Research also shows EEG only carries coarse information ([Beyond Reconstruction](https://arxiv.org/pdf/2505.21385)).
- **Treating EEG as a video** for self-supervised video models ([EEG-VJEPA](https://arxiv.org/abs/2507.03633v2)): too experimental.

---

## 5. Suggested plan for a second track
1. **Option B (relationship matching):** quick to build, independent of the proven track, and naturally robust across people. A good "completely different" solution.
2. **Option A (time-staged experts):** strongest scientific story, and it reuses the cached features from the proven track.
3. **Option E (MVNN, intermediate layers):** try on both tracks.
4. **Final step:** combine the scores of both tracks (Option D lesson).
5. **Questions for the organizers:** (a) Are BMD annotations or fMRI for the videos allowed? (b) Is one-to-one assignment at test time allowed?
