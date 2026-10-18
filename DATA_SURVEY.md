# Data survey: public data and models beyond the challenge set

Written 2026-10-18 by krabiki-51. Research only: no data downloaded, no GPU used. Facts marked **(unverified)** were not
confirmed on a primary page; licenses and sizes are as stated on the pages linked below and should be re-checked before use.

**Update (later on 2026-10-18):** the organizers gave our 17 channel names and the preprocessing; they are exactly the THINGS-EEG2 17-channel posterior set, which removes the montage blocker (§6, new §8).

Scope set by the user (via krabiki-e1): **every source built on our stimulus set is out of scope** (the same 1,102 videos, or the 102 test videos). Rule recap
(ClaudeREADME §3): external models and datasets must be public and released before 2026-10-16, with exact name and version recorded; never train on test data.

## 0. Verified facts about our stimuli (local)

- 1,102 files: `stimuli_train/0001..1000.mp4` + `stimuli_test/1001..1102.mp4`; all 268 × 268; 3.0 s (1,094 clips) or 3.1 s (8 clips); frame rates 30 (424), 29.97 (390), 25 (82),
  24 (71), 23.98 (69), a few others; h264 + AAC.
- **Audio is present in 1,068 of 1,102 clips** (34 clips have no audio stream). Sample rates 32/48 kHz, stereo.
- This matches the published description of the Algonauts 2021 / BOLD Moments Dataset stimulus set (1,102 clips, 1,000 + 102, 3 s, square-cropped and resized to 268 × 268,
  15-30 fps; clips from Memento10k, a subset of Moments in Time). The fMRI experiment showed them silent; ours have sound. I did not compare frames or hashes, since that would need the
  BMD stimulus download, which is out of scope.

## 1. EXCLUDED: sources built on our stimulus set (do not download, open or use)

| Source | Why excluded |
|---|---|
| **EEG Moments Dataset (EMD)**, OpenNeuro ds008257 ([page](https://github.com/OpenNeuroDatasets/ds008257), [code](https://github.com/gifale95/EMD), [mirror](https://github.com/bruAristimunha/on008257)); paper arXiv 2608.28768 | **HARD STOP.** EEG for the same 1,102 videos, 6 subjects, 1,000 videos × 6 repeats + 102 test videos × 24 repeats. Very likely contains this challenge's recordings and the test-video ↔ recording mapping. A search snippet describes it as **128-channel** with eye tracking, released Aug-Sep 2026. If our 17 channels are a subset of its montage, it would also solve our montage problem. That is exactly why it must stay unopened. The user is informing the organizers. |
| BOLD Moments Dataset fMRI, OpenNeuro ds005165 ([page](https://github.com/OpenNeuroDatasets/ds005165)) and its stimulus/metadata package | Same videos. fMRI responses (10 subjects, test videos × 10 repeats), `annotations.json` (15 object labels, 5 scene, 5 action, 5 sentences, transcription, memorability), TSM checkpoints trained on a subset of Multi-Moments in Time. |
| Algonauts 2021 data ([challenge](https://algonauts.csail.mit.edu/2021/challenge.html)) | Same videos and fMRI. |
| Moments in Time / Multi-Moments in Time / Spoken Moments in Time ([site](http://moments.csail.mit.edu/)), Memento10k | Source collections of the clips; annotation sets for them cover our videos. Access also needs a request form (academic use). |
| **TRIBE v2** brain-encoding model ([facebook/tribev2](https://huggingface.co/facebook/tribev2), [paper](https://arxiv.org/abs/2605.04326)) | A general pretrained model, but its training set (Table 1 of the paper) **includes BoldMoments (10 subjects, 61.9 h)**, i.e. fMRI responses to our test clips. CC BY-NC 4.0. Not usable under the narrowed scope. |
| Models trained on Moments in Time (e.g. [MiT TSM models](https://github.com/zhoubolei/moments_models)) | Gray area: general public models, but trained on a collection that contains our clips. Default: avoid; prefer models trained on other data (Kinetics, web). Ask if wanted. |

## 2. Ranked shortlist (value × feasibility × rule risk)

Hackathon work days are Oct 17-18 and today is Oct 18, so anything needing more than a few hours is unrealistic.

| Rank | Idea | Value | Feasibility today | Rule risk | Notes |
|---|---|---|---|---|---|
| 1 | **Self-computed video-level features** with a public video encoder (V-JEPA 2) | medium-high | high (~1 h) | very low | New *temporal/motion* signal; M6 found frames/pooling of image features closed. See §3.1. **On hold** until the user confirms the `transformers` install. |
| 2 | ~~THINGS-EEG2 transfer~~ **tested, no-go** (§8.4) | low (weak zero-shot signal) | done | low | Linear zero-shot transfer is real but ~2x chance at top-1; no dev gain as ensemble member or prior. |
| 3 | **Self-generated captions** → text embeddings as extra decoding targets | medium | high (~1-1.5 h) | very low | See §3.2. **On hold.** |
| 4 | **Audio embeddings of our own soundtracks** (CLAP etc.) | medium | high (~45 min) | very low | See §3.3. **On hold** (user deferred audio). |
| 5 | Frozen EEG-foundation-model embeddings (CBraMod / REVE / LaBraM) as extra EEG-side features | low-medium | medium (electrode names are known now) | low (license checks) | Clinical pretraining; see §5. |
| 6 | Other EEG image datasets (Alljoined-1.6M, EEG-ImageNet, THINGS-EEG1) | low | low | low (Alljoined is CC BY-NC-SA) | Different channel sets; only after rank 2 shows transfer works. |
| 7 | SEED-DV (video paradigm), HBN-EEG | low | low (application form, 62-128 ch) | low | See §4. |

**Concrete next steps:** (1) the THINGS-EEG2 linear go/no-go test of §8 (needs a download approval for 2-3 subjects, 2-3 GB); (2) once the user confirms the `transformers` install: cache V-JEPA 2 video features for the 1,102 clips and test `decw:vjepa2_*` alone and as an ensemble member with `scripts/ensemble.py`, then Qwen2-VL-2B or SmolVLM2 captions with the same evaluation.

## 3. Direction A, clean version: stimulus features we compute ourselves

These are the same status as our CLIP/DINOv2 features: frozen public models applied to our own videos. Nothing about the clips is downloaded.

### 3.1 Video encoder (top pick)
- **V-JEPA 2** (Meta FAIR, released 2025-06-25, MIT license): HF [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/collections/facebook/v-jepa-2) (ViT-L, ~300M params; also ViT-H and ViT-g; Transformers support). Self-supervised on video, so it carries
  motion/dynamics that frame-wise CLIP/DINOv2 lack. A V-JEPA 2.1 followup (2026-03-16) exists **(not checked)**.
- Plan: 64 frames per clip sampled uniformly over the 3 s at 256 px (the ViT-L model expects 64-frame clips; the SSv2 fine-tune uses 16 frames), take the pooled last-layer tokens (1,024-d)
  plus optionally the mean of an intermediate layer. Cache as `features/vjepa2_l_all_f1.npz` (ids, mean) so `--member decw:vjepa2_l_all:1` works. Estimated 5-15 min of GPU (**estimate, not measured**). 8 GB should suffice at batch size 1
  in fp16. New dependency: `transformers` (not currently installed; PyPI is reachable).
- Alternatives: VideoMAE / X-CLIP (older, Kinetics-trained) **(not checked)**.

### 3.2 Captions → text embeddings
- Idea: caption each clip with a public video-language model, embed the caption with a text encoder (OpenCLIP ViT-B-32 laion2b text tower, or a sentence encoder), and use it as a decoding target / ensemble member.
- Candidate captioners that fit an 8 GB GPU (all checked on their HF cards):
  | Model | License | Params | Video | Note |
  |---|---|---|---|---|
  | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Apache-2.0 | 2B (bf16) | yes (frame list or file, fps setting; no audio) | Needs recent `transformers` + `qwen-vl-utils`; cap tokens with `max_pixels`. Third-party VRAM estimate ~5 GB fp16 |
  | [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | Apache-2.0 | 2.2B | yes | Card states 5.2 GB GPU RAM for video inference |
  | [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | MIT | 0.77B | image only (use 1-8 frames) | `<MORE_DETAILED_CAPTION>`; `trust_remote_code=True` |
  | Qwen2.5-VL-3B / Qwen2.5-Omni-3B | check | 3-4B | yes (Omni also audio) | Omni needs ~18 GB in bf16 per the official table, so **does not fit**; 3B VL is ~8 GB fp16 (tight) |
- Cost (**estimate**): a 2B video-language model at 8 frames/224 px, ~40 generated tokens: ~1-3 s per clip → 20-60 min for 1,102 clips; Florence-2 is faster but image-only.
- Expected value: moderate. Captions are produced from the same pixels CLIP sees, so most of the gain will be the action/verb information. Use several captions per clip (different prompts) and average the embeddings to reduce captioner noise.
- Caveat: captions of the *test* videos are our own features, like CLIP image features, so they are allowed by the rules as the user ruled; keep generation deterministic and unsupervised (no test EEG).

### 3.3 Audio (needs a go-ahead; deferred earlier)
- LAION-CLAP ([repo](https://github.com/LAION-AI/CLAP), `laion-clap` 1.1.7 on PyPI, checkpoints released 2022-11 and 2023-04; HF [`laion/larger_clap_music_and_speech`](https://huggingface.co/laion/larger_clap_music_and_speech) lists apache-2.0) or other audio encoders **(not checked)**.
  Extract the audio track with ffmpeg (48 kHz), embed, then use as a decoder target or ensemble member. 34 of 1,102 clips have no audio: embed silence or zero-fill and report it.
- Value: audio-evoked cortical responses (N1/P2 around 100-200 ms) are part of our EEG and unexplained by vision features. Our clips have sound; the matching fMRI stimuli did not.

## 4. Direction B: public EEG datasets for pretraining (different stimuli)

| Dataset | Subjects | Trials / stimuli | Channels | Rate | Stimulus | Size | License | Released |
|---|---|---|---|---|---|---|---|---|
| **THINGS-EEG2** (Gifford et al. 2022; [OSF](https://osf.io/3jk45/), [NEMAR nm000232](https://nemar.org/dataset/nm000232)) | 10 | 82,160 trials: 16,540 train images × 4 + 200 test × 80 | 63 (10-10, BrainVision actiCHamp) | 1000 Hz | THINGS images, RSVP (5 Hz) | **241 GB** raw BIDS (preprocessed derivatives smaller, **unverified**) | CC-BY 4.0 | 2022 (BIDS mirror versions in 2026) |
| **THINGS-EEG1** (Grootswagers et al. 2022; [OpenNeuro ds003825](https://openneuro.org/datasets/ds003825)) | 50 | 22,248 images / 1,854 concepts, 1 exposure each (+200 repeated 12×) | 64 **(unverified)** | **(unverified)** | THINGS images, RSVP 10 Hz | 41 GB | CC-BY 4.0 | Jan 2022 |
| **Alljoined-1.6M** ([HF](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M), [paper](https://arxiv.org/abs/2508.18571)) | 20 | ~1.69M trials, 16,740 images | 32 (consumer EMOTIV FLEX2, extended 10-20) | 256 Hz | THINGS images, RSVP | 136 GB | **CC-BY-NC-SA 4.0** | Aug 2025 |
| **Alljoined1** ([OSF](https://osf.io/kqgs8/), [arXiv](https://arxiv.org/abs/2404.05553)) | 8 | 46,080 epochs, 10,000 COCO images each | 64 **(unverified)** | **(unverified)** | COCO images | ? | not specified | 2024 |
| **EEG-ImageNet** ([repo](https://github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset), [arXiv](https://arxiv.org/abs/2406.07151)) | 16 | 4,000 images (80 classes) per subject; 0.5 s per image | 62 | 1000 Hz | ImageNet images | ? (two .pth files) | MIT | 2024 (date not stated) |
| **SEED-DV / EEG2Video** ([BCMI-SJTU](https://bcmi.sjtu.edu.cn/home/eeg2video/), NeurIPS 2024) | 20 | 1,400 natural video clips, 40 concepts | 62 | 200 Hz (shared release) | **videos** (clip length not stated) | ? | application form + license agreement (academic) | 2024 |
| **HBN-EEG** ([paper](https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2.full), [challenge](https://eeg2025.github.io/eeg2025.github.io/)) | 3,000+ | 4 continuous movies (Despicable Me etc.) plus 5 other tasks | 128 (EGI) | 500 Hz | continuous movies, no per-clip labels | multi-TB | see source | 2024-2025 |
| PAIR ([arXiv 2610.13120](https://arxiv.org/abs/2610.13120)) | 20 | naturalistic videos + visual recall | ? | ? | videos | ? | ? | submitted 2026-10-09; **no data link found, release date unverified (may be after the cutoff)** |

None of these contains our stimuli, as far as the dataset descriptions show (the video ones, SEED-DV and PAIR, use their own clips; PAIR's source of clips is unverified).

**Assessment.** All image datasets are RSVP paradigms (100-200 ms per image), with 63-64 research-grade channels at 1000 Hz (or a 32-channel consumer system), versus our 3-s videos with sound, 17 unnamed channels at 500 Hz. Transfer would at best teach a generic early-visual ERP front end (50-300 ms),
which our encoder already learns from ~900 trials per subject. I found **no published case** of pretraining on THINGS-EEG2 and fine-tuning on a different montage for retrieval (the closest, ENIGMA, trains one model across THINGS-EEG2 and Alljoined-1.6M jointly), and
a heterogeneous-electrode GNN paper (motor imagery, [arXiv 2306.13109](https://arxiv.org/html/2306.13109)) fine-tunes on the shared channels only. Pretraining can also fail to help ([arXiv 1806.09532](https://arxiv.org/pdf/1806.09532)).
Practical blockers: THINGS-EEG2 raw is 241 GB against 174 GB free disk, but the *preprocessed* 17-channel release is only ~10.7 GB (§8), so this is now feasible with an approved download; GPU time competes with Stage 3. The montage problem is solved by the organizers' channel list (§6).
SEED-DV is the closest paradigm (video watching) but needs an application and approval time, so it is not usable in this hackathon.

## 5. Direction C: EEG foundation models

| Model | Venue / date | Pretraining | Channel handling | Rate | Weights / license |
|---|---|---|---|---|---|
| **LaBraM** ([repo](https://github.com/935963004/LaBraM)) | ICLR 2024 | ~2,500 h, ~20 datasets | spatial embedding looked up **by electrode name** (10-20 vocabulary), any subset/order; pretrained on dense >120-channel configs | 200 Hz **(unverified)** | repo MIT; weights in repo / HF via braindecode; weight license not stated |
| **EEGPT** | NeurIPS 2024 | 246 h, 5 datasets | channel embedding "codex", up to 58 channels | 256 Hz **(unverified)** | not checked |
| **BIOT** | NeurIPS 2023 | clinical, limited montages | tokenizes channels separately | - | not checked |
| **CBraMod** ([repo](https://github.com/wjq-learning/CBraMod), [arXiv](https://arxiv.org/abs/2412.07236)) | ICLR 2025 | TUEG, ~27,000 h clinical, >40 configs | criss-cross spatial/temporal attention + convolutional positional encoding, **no electrode names needed** | 200 Hz, 30 s × 1 s patches | weights in repo and on HF; ~4-5M params; weight license **not confirmed** (TUEG terms may apply) |
| **REVE** ([HF](https://huggingface.co/brain-bzh/reve-base), NeurIPS 2025) | Oct 2025 | >60,000 h, 92 datasets, 25,000 subjects | **3D electrode coordinates** from a position bank (`brain-bzh/reve-positions`), any layout | 200 Hz | gated; "REVE Responsible Use License"; base 69M params |
| **LUNA** ([arXiv 2510.22257](https://arxiv.org/html/2510.22257)) | NeurIPS 2025 | TUEG + Siena, ~21,900 h | topology-agnostic (3D coordinates + learned queries) | 256 Hz | code link given; the checklist says code/weights "will be published upon publication": **availability unverified** |

**Assessment.** All were pretrained mainly on clinical/resting EEG for classification tasks (sleep, seizures, abnormality, emotion, motor imagery); none targets stimulus-locked visual retrieval. Most need electrode names or 3D coordinates; **we now have the names** (and standard 10-05 coordinates from them), so LaBraM and REVE are usable on these 17 posterior channels, although they were pretrained mostly with all-scalp montages. A benchmark found
[channel adaptation matters and native-flexible models (EEGPT, LUNA, CBraMod) match external adapters](https://arxiv.org/html/2604.23091v1); LUNA reports a weaker result on a new 62-channel montage. Realistic use: frozen embeddings per 1-s patch (resampled to 200 Hz) as additional EEG input features to the decoder. Expected gain small; skip unless time remains.

## 6. The montage problem (resolved by the organizers' channel list; the probe below is kept as supporting evidence)

**Organizers (relayed by krabiki-e1):** our 17 channels, in order and the same for all subjects, are `Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2`; already applied: band-pass 0.1-40 Hz, blink components removed, downsampled 1000 → 500 Hz.
This is, as a set, exactly the 17 posterior channels of THINGS-EEG2 (`O1 Oz O2 PO7 PO3 POz PO4 PO8 P7 P5 P3 P1 Pz P2 P4 P6 P8`, checked in code; only the order differs).
**The data agree with the stated order:** with the standard 10-05 electrode positions (MNE `standard_1005`), the Spearman correlation between inter-electrode distance and our filtered (1-30 Hz, common-average) channel correlation is -0.84, -0.85, -0.81, -0.86, -0.85, -0.85 for sub-01..06; none of 2,000 random channel orderings reaches that (p < 0.0005 each).

### Earlier blind probe on our training EEG (before the channel list was known)

Cheap probe on the 6 subjects' training recordings (decimated to 100 Hz):

- **All subjects share the same channel layout.** After 1-30 Hz band-pass and common-average referencing, the 17 × 17 channel-correlation matrices agree across subjects with Spearman ρ = 0.97 (min 0.95; 0.83 before filtering, where a common component holds 84% of the variance).
- The spatial structure is low-dimensional: two principal components hold 78% of the filtered variance (44% + 34%), the next ones 9%, 3%, 2%.
- The channel-wise evoked (trial-mean) response patterns are only weakly consistent across subjects (mean pattern correlation 0.17, range -0.32 to 0.56); the nearest-neighbour partner of a channel is identical in only 48% of subject pairs. So neighbour structure is noisy.
- Per-channel std is ~0.48 for every channel and subject (already normalized), so no amplitude cue.

Approaches, by realism for a hackathon:

| Approach | Realism | Comment |
|---|---|---|
| **Channel-agnostic encoder trained on our data only** (shared temporal front end per channel + learned channel embedding / spatial mixing; the existing ridge models are already channel-position-free) | high | No names needed. This is what we effectively do now. |
| Learned coordinates: initialize 2-D channel coordinates by MDS of the correlation distances (above), refine by training | medium | Gives a spatial prior for REVE/LUNA-style position encodings without labels. No ground truth to validate against. |
| Template matching: assign our channels to 10-20 labels by matching channel-correlation (or alpha/blink topography) structure to a template computed from a *different* public dataset with known positions (THINGS-EEG2, 63 channels) via a quadratic assignment | low | 17 of 63 positions: many near-equivalent assignments; blink (frontal) and alpha (posterior) cues might fix coarse anterior-posterior / left-right order only. Requires downloading and preprocessing a large dataset. Not reliable enough for a name-based model such as LaBraM. |
| Solve the layout from the 128-channel EEG of the same experiment (EMD) | **forbidden** | Excluded in §1; no longer needed. |

Net (superseded): without names, exact labels were not recoverable; with the organizers' list, §8 applies and name-based models (LaBraM, REVE) become usable too.

## 7. Recommendations

1. **Do now (rank 1 and 2, ~2 h total):** V-JEPA 2 video features and self-generated captions, cached in `features/` under the existing naming, evaluated as `decw` / ensemble members with the N-member CV tool. Both add information types the current CLIP/DINOv2 members do not have (motion, language).
2. **Needs a go-ahead:** audio embeddings (rank 3); any large download (e.g. THINGS-EEG2 preprocessed subset), since disk has 174 GB free.
3. **Park:** EEG foundation-model embeddings. EEG pretraining is no longer parked: see rank 2 and §8.
4. **Do not touch:** everything in §1. TRIBE v2 looks attractive but was trained on BoldMoments.

## 8. THINGS-EEG2 preprocessed release: details and a minimal transfer plan

### 8.1 What the release is (verified from the OSF API, the preprint and the NICE / ATM / Gifford code)
- **Where:** OSF project [3jk45](https://osf.io/3jk45/), component [`anp5v` "Preprocessed EEG data"](https://api.osf.io/v2/nodes/anp5v/) (created 2021-06-24, last modified 2026-04-07, license **CC BY 4.0**). The paper (Gifford, Dwivedi, Roig, Cichy; preprint 2022-03, NeuroImage 2022) says the raw and preprocessed EEG, the image set and DNN features are on OSF.
- **Files in `anp5v`:** `sub-01.zip … sub-10.zip`, **~1.07 GB each (10.7 GB total, uploaded Aug 2022)**, and `sub-XX__63_channels.zip`, ~4.03 GB each (40.4 GB, uploaded Apr 2026). Zip contents were not opened. From the paper's description and the size, the 1.07 GB files are the 17-channel version: `16,540 × 4 × 17 × 100` + `200 × 80 × 17 × 100` float64 values ≈ 1.1 GB (my arithmetic).
- **17 channels:** `O1 Oz O2 PO7 PO3 POz PO4 PO8 P7 P5 P3 P1 Pz P2 P4 P6 P8` (preprint). Gifford's own preprocessing script selects channels by the regex `^O *|^P *` (occipital and parietal names), while the NICE/ATM scripts keep all 63 (their occipital/parietal pick is commented out); in both cases the stored `ch_names` allow selecting the 17 by name.
- **Preprocessing of the Gifford release (preprint + code):** recorded at 1000 Hz with online 0.1-100 Hz filter, online reference **Fz**, 64-ch actiCHamp; MNE offline: epochs **-200 to 800 ms** around image onset, baseline = mean of the pre-stimulus interval per trial and channel, **downsampled to 100 Hz (100 time points)**, target-image trials removed, repetitions subsampled to 4 (train) / 80 (test), **multivariate noise normalization (MVNN, Ledoit-Wolf-type shrinkage covariance) per session**. No offline band-pass filter and no blink removal are described.
- **Shapes per subject:** train 16,540 conditions × 4 reps × 17 ch × 100 samples; test 200 × 80 × 17 × 100. 10 subjects, 4 sessions each, 82,160 trials per subject (821,600 in total).
- **Paradigm:** images (THINGS) 7° visual angle, **100 ms on, 200 ms SOA (5 Hz RSVP)**, sequences of 20; fixation task (detect a Buzz Lightyear target). So a trial's response is contaminated by the next image after ~200 ms; the clean part is the first ~200 ms.
- **Images:** component [`y63gw`](https://api.osf.io/v2/nodes/y63gw/files/osfstorage/): `training_images.zip` 655 MB, `test_images.zip` 8 MB, `image_metadata.npy`, a `LICENSE.txt` (not read). A Hugging Face mirror for ATM ([LidongYang/EEG_Image_decode](https://huggingface.co/datasets/LidongYang/EEG_Image_decode), CC-BY-4.0) also has `images_set.tar.gz` (778 MB) and CLIP ViT-H-14 features.
- **NICE ([repo](https://github.com/eeyhsong/NICE-EEG), MIT) and ATM ([repo](https://github.com/dongyangli-del/EEG_Image_decode), MIT, NeurIPS 2024) preprocessing:** same script family: **all 63 channels**, epoch -0.2 to 1.0 s, baseline on the pre-stimulus part, **resampled to 250 Hz** and cropped to 0-1.0 s (250 samples), MVNN per session with the covariance estimated on the training partition only, **no filtering**; 2 repetitions per session kept for training and 20 for test; files `Preprocessed_data_250Hz/sub-XX/preprocessed_eeg_{training,test}.npy` (pickled dicts with `preprocessed_eeg_data`, `ch_names`, `times`). They are therefore not 17-channel and not 100 Hz; the Gifford release is the match for us.
- **Release date:** OSF component created 2021-06; preprint 2022-03; preprocessed 17-channel zips uploaded 2022-08-12; the 63-channel zips 2026-04-07. All before 2026-10-16.

### 8.2 Alignment with our data
| | THINGS-EEG2 (Gifford release) | Ours (organizers) | Alignment |
|---|---|---|---|
| Channels | 17 posterior, names above | same 17, different order | reorder by name to `Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2` |
| Rate | 100 Hz | 500 Hz | our pipeline already decimates to 100 Hz |
| Filter | online 0.1-100 Hz, no offline filter | 0.1-40 Hz | low-pass THINGS at 40 Hz (zero-phase) |
| Reference | online Fz | **unknown**; our channels share a large common component before filtering | ask the organizers; common-average re-reference both as a fallback |
| Blinks | not removed (posterior channels are mildly affected) | removed | accept |
| Baseline | per trial on the 200 ms before onset | no pre-stimulus segment in our 3-s recordings | z-score each trial over the 0-600 ms window for both (correlation matching already ignores per-trial offset/scale) |
| Whitening | MVNN per session (already applied to the release) | none (ours); M5/M8 use Ledoit-Wolf noise whitening | apply our training-only noise whitening to our side so that both are whitened |
| Epoch | -200..800 ms; the useful, clean part is 0-200 ms (RSVP overlap) | 0-3 s, we match on 50-600 ms | use THINGS samples 25:80 (50-600 ms) as the aligned window; expect the first ~200 ms to carry most of the transferable signal |
| Stimulus | static object images, 100 ms | 3-s dynamic scenes with sound | the main domain gap |

### 8.3 Minimal Stage-3 plan (research-only; every download below needs approval)
**Step 0, linear go/no-go (~1.5 h; downloads 2-3 subjects' `sub-XX.zip` = 2-3 GB plus `training_images.zip`/`test_images.zip` = 0.66 GB):**
1. Embed the 16,740 THINGS images with the same CLIP B-32 (laion2b_s34b_b79k) image tower we use for the videos (~5-10 min GPU).
2. Load the 17-channel arrays, reorder, low-pass 40 Hz, take samples 25:80, z-score, average the 4 training repetitions per condition (cleaner targets) or keep all; train the whitened decoder (our `decw` ridge, EEG window → standardized CLIP vector) on the THINGS subjects.
3. Apply it to our EEG **as is**, zero-shot across datasets and subjects (our EEG whitened and z-scored the same way), score the 102 candidates with the cosine to the video CLIP vectors, on sub-01..04 dev. Compare with chance (1.0% / 4.9% top-1/top-5) and with our own per-subject decw (14.7% / 42.2%).
4. **Kill criterion:** if cross-dataset zero-shot top-1 is within noise of chance (< ~2.5% on the 408 dev trials, where the standard error is about 0.5 pts at chance), stop; there is no usable shared signal. If it is clearly above chance, also test it as an ensemble member (`ensemble.py`) and as a prior for the ridge solution (shrink the per-subject ridge towards the THINGS solution, which is a cheap form of transfer).

**Step 1, neural pretraining (only if Step 0 passes; ~4-6 h):** a small spatio-temporal encoder (NICE/EEGNet-style temporal conv, spatial conv across the 17 channels, linear projection to 512-d) trained with InfoNCE against the CLIP image embeddings on all 10 THINGS subjects (~820k trials, 17 × 55-80 samples; ~3-4.5 GB as float32 after cropping, ~2 GB as float16, fits in RAM). Use a held-out THINGS subject for early stopping. GPU: ~15-60 min on the 8 GB card, shared with krabiki-2f's Stage 3.

**Step 2, transfer to our per-subject decoders (~1-2 h):** (a) use the pretrained encoder's 512-d output as extra features in `decw`-style ridge, or as a `con`-type member (EEG embedding, cosine to the video CLIP vector); (b) fine-tune the last layers with InfoNCE on each subject's training trials (small learning rate, early stopping on an inner split, the M12 recipe); choose by the training-only CV and the existing LOFO rule. Evaluate with the same dev table and Wilcoxon.

**Expected value, honestly:** the pretraining volume is ~150× our trials, but the stimuli are single objects shown for 100 ms in an RSVP stream, not 3-s scenes with motion and sound, and single-trial decoding on THINGS-EEG2 is far weaker than the 80-repeat-averaged numbers in the papers. I would expect at most a small gain (order +0-1.5 pts top-1 as a blend member) and a real chance of none; the go/no-go test is cheap precisely to settle that. **Effort in total: ~1.5 h for Step 0, ~6-9 h for the full chain.**

**Rule check:** THINGS-EEG2 has different stimuli and subjects (CC BY 4.0, public since 2021-2022), so the rules in §3 are met, and nothing here touches our test recordings except for final scoring. Record the exact source: OSF `3jk45/anp5v`, files `sub-XX.zip`, Gifford et al., NeuroImage 2022.

### 8.4 Step 0 executed (2026-10-18): data record and result
**Data record (rule 2).** Source: THINGS-EEG2 preprocessed 17-channel release, Gifford, Dwivedi, Roig & Cichy, "A large and rich EEG dataset for modeling human visual object recognition", NeuroImage 2022 (preprint bioRxiv 2022-03). OSF project `3jk45`:
component `anp5v` "Preprocessed EEG data" (CC BY 4.0): `sub-01.zip` (https://osf.io/download/nb8wr/) and `sub-02.zip` (https://osf.io/download/u43p5/), 1,071,722,467 and 1,071,516,660 bytes, uploaded 2022-08-12, version 1;
component `y63gw` "Image set": `training_images.zip` (https://osf.io/download/3v527/), `test_images.zip` (https://osf.io/download/znu7b/), `image_metadata.npy` (https://osf.io/download/qkgtf/), `LICENSE.txt` (https://osf.io/download/3765v/); THINGS images are for research use only, no redistribution.
sha256 of all files matched the values published by the OSF API. Stored in `/home/kparpie/ext_data/things_eeg2/` (outside every repo). Zip content confirmed: `sub-XX/preprocessed_eeg_training.npy` is 899,777,433 bytes = 16,540 x 4 x 17 x 100 float64 values + header; `preprocessed_eeg_test.npy` 217,601,432 bytes = 200 x 80 x 17 x 100.
The pickled dicts were loaded once, in isolated mode (`python -I`), with a whitelisting unpickler (numpy array reconstruction and dict/list only; nothing was refused) and re-saved as plain float32 `.npz`. **The stored channel order is exactly our organizer-given order** (`Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2`), times -0.20..0.79 s at 100 Hz.
Image embeddings: `open_clip ViT-B-32 / laion2b_s34b_b79k` (same as our video features).

**Result (details: PROGRESS.md "THINGS-EEG2 Step 0", MODEL_BACKLOG M15).** Within THINGS the linear map works (200-way single-trial top-1 3.4%, 80-rep average 8%). Zero-shot on our EEG there is a weak but real shared signal: pooled over the 6 subjects' training
recordings the mean percentile rank is 0.43 (chance 0.5), E[top-1] ~2% (chance 1%), E[top-5] ~8% (chance 4.9%); on the dev test sets top-1 is 0.7-2.2% (noise of chance), top-5 6-9%. As an ensemble member the CV gives it weight 0.2 (LOFO +0.005) but dev shows no gain (14.5 / 40.7 vs 14.7 / 42.2, p = 0.84);
as a shrinkage prior the CV gain is 0.0016 in percentile rank (6/6 subjects, negligible). **Verdict: no-go for neural pretraining (Steps 1-2).** The rank-2 entry in §2 is superseded by this result: the transfer is too weak, the domain gap (static objects at 5 Hz RSVP vs 3-s scenes with sound) is the limit.

## Sources (accessed 2026-10-18)

[BOLD Moments Dataset paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11269733/) · [OpenNeuro ds005165](https://github.com/OpenNeuroDatasets/ds005165) · [BMD starter code](https://github.com/blahner/BOLDMomentsDataset) · [Algonauts 2021](https://algonauts.csail.mit.edu/2021/challenge.html) ·
[Moments in Time](http://moments.csail.mit.edu/) · [Spoken Moments in Time](http://moments.csail.mit.edu/spoken.html) · [TRIBE v2 paper](https://arxiv.org/abs/2605.04326) / [model card](https://huggingface.co/facebook/tribev2) · [THINGS-EEG2 (NEMAR)](https://nemar.org/dataset/nm000232) · [THINGS-EEG1 (OpenNeuro ds003825)](https://openneuro.org/datasets/ds003825) ·
[Alljoined-1.6M (HF)](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M) · [EEG-ImageNet](https://github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset) · [EEG2Video / SEED-DV](https://bcmi.sjtu.edu.cn/home/eeg2video/) · [HBN-EEG](https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2.full) · [PAIR](https://arxiv.org/abs/2610.13120) ·
[LUNA](https://arxiv.org/html/2510.22257) · [REVE (HF)](https://huggingface.co/brain-bzh/reve-base) · [CBraMod](https://github.com/wjq-learning/CBraMod) · [LaBraM](https://github.com/935963004/LaBraM) · [channel-adaptation benchmark](https://arxiv.org/html/2604.23091v1) ·
[THINGS-EEG2 preprint](https://www.biorxiv.org/content/10.1101/2022.03.15.484473v1.full) · [OSF preprocessed component anp5v](https://api.osf.io/v2/nodes/anp5v/files/osfstorage/) · [NICE-EEG](https://github.com/eeyhsong/NICE-EEG) · [ATM / EEG_Image_decode](https://github.com/dongyangli-del/EEG_Image_decode) · [Gifford eeg_encoding](https://github.com/gifale95/eeg_encoding) · [V-JEPA 2](https://github.com/facebookresearch/vjepa2) · [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) · [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) · [Florence-2](https://huggingface.co/microsoft/Florence-2-large) · [Qwen2.5-Omni](https://github.com/qwenlm/qwen2.5-omni) · [LAION-CLAP](https://github.com/LAION-AI/CLAP).

## 9. Video-EEG candidates (krabiki-cc)

Written 2026-10-18 by krabiki-cc (the `track-viz` / `track-vjepa` session). Research only: web search and page reads, plus HF metadata calls; **nothing was downloaded and no data file was opened**. Facts marked **(unverified)**
were not confirmed on a primary page. Scope: public EEG datasets with stimuli closer to ours (seconds-long natural video clips, stimulus-locked trials), released before 2026-10-16.
The lesson from §8.4 is the yardstick: the transfer blocker was the stimulus domain, not channels or data size.

### 9.1 EXCLUDED (links only; not opened, not used)

| Source | Why excluded |
|---|---|
| EEG Moments Dataset (EMD), [OpenNeuro ds008257](https://github.com/OpenNeuroDatasets/ds008257), [arXiv 2608.28768](https://arxiv.org/abs/2608.28768) | Already in §1: same 1,102 videos. A search result surfaced its abstract; I did not open the page or the data. |
| **AVMoments-EEG** ([Zenodo 10.5281/zenodo.22227356](https://zenodo.org/records/22227356), [code](https://github.com/ShuningTangg/AVMoments-EEG)) | **Built on the Moments-in-Time family.** Its README says the 3-s stimuli are "derived from the Audiovisual Moments in Time dataset" ([AVMIT](https://arxiv.org/html/2308.09685)), an annotated subset of Moments in Time, which is the source collection of our clips (§1). I cannot tell whether any clip is shared with ours (that would need comparing against the stimulus set), so I treat it as excluded under the family rule; **ask krabiki-e1 / the user if a ruling is wanted**. It would also be unusable as is: the videos are withheld (licensing; request by e-mail to the authors), so no CLIP/DINOv2/V-JEPA features could be computed without a request. Otherwise it is the most similar paradigm found: CC BY 4.0, 10 subjects, 64 ch (ANT eego, extended 10-10), 1000 Hz, resampled to 100 Hz, 960 clips (896 × 3 repeats + 64 × 40 repeats), 10,368 trials per subject, 16 audiovisual event classes. |
| BOLD Moments (fMRI), Algonauts 2021, TRIBE v2 | As in §1. |

### 9.2 Candidates that passed the filters, with their problems

| | **SEED-DV / EEG2Video** | **CineBrain** |
|---|---|---|
| Links | [project](https://bcmi.sjtu.edu.cn/home/eeg2video/), [code](https://github.com/XuanhaoLiu/EEG2Video), [NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/84bad835faaf48f24d990072bb5b80ee-Paper-Conference.pdf) | [HF Fudan-fMRI/CineBrain](https://huggingface.co/datasets/Fudan-fMRI/CineBrain), [arXiv 2503.06940](https://arxiv.org/abs/2503.06940), [code](https://github.com/yanweifu-sii/CineBrain) |
| Released | 2024-12-14 (GitHub news) | HF repo created 2025-03-15, EEG files uploaded from 2025-04-08, last modified 2026-04-07 (HF metadata) |
| Access / licence | **Application plus a signed licence file** (academic, via SJTU BCMI); terms not seen. Whether a gated release counts as "public" under rule 3 is a question for the organizers. | **Open, not gated**, Apache-2.0 (HF card). The clips are copyrighted TV footage, and the card says nothing about their terms. |
| Stimuli | 1,400 natural clips of **2 s**, 40 concepts × 35, "collected from the Internet" (slides); **included in the download** (7 block videos), with BLIP captions. Audio: **(unverified)**. | 8,100 four-second clips of *The Big Bang Theory* (18 min of each of 30 episodes), `videos.tar` 2.6 GB; audio is hosted separately (ModelScope). Continuous sitcom with dialogue. |
| Subjects / trials | 20 subjects × 1,400 clips = 28,000 single-view trials (each clip seen once per subject). | 6 subjects × 5,400 clips (about 6 h each). 4-s clips are consecutive, not separate trials. |
| Design | Concept blocks: a 3-s text hint, then **5 clips of the same concept back to back**, 7 blocks with rests. So only the first clip of each group is a clean onset response (280 per subject), and the others carry the previous clip's response. | Continuous viewing. There is no stimulus-onset event: EEG is cut into 0.8-s windows (5 per clip), aligned to fMRI TRs. Our signal is an onset-locked 50-600 ms evoked response, so this paradigm is the least like ours. |
| EEG | ESI NeuroScan, **62 ch**, 1000 Hz recorded, **200 Hz released**, raw. The standard SEED layout has P7 P5 P3 P1 PZ P2 P4 P6 P8 / PO7 PO5 PO3 POZ PO4 PO6 PO8 / CB1 O1 OZ O2 CB2 (from a layout excerpt, [SEED page](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html)). **So all 17 of our channels are present, by name** (the channel-order file in the download should be checked). | 64-ch MR-compatible cap in the scanner, **EGI GSN-HydroCel-64 layout (E1-E64, not 10-10 names)**, 1000 Hz, preprocessed (scanner/ECG artifacts removed by QRS + ICA, 0.1-30 Hz band-pass, 50 Hz notch; reference not stated). **Our 17 sites need a nearest-electrode mapping, and several (PO7/PO8, P5/P6, P1/P2, POz) probably have no exact match (unverified).** |
| Size | **Not stated**; my estimate 3-6 GB for the EEG (20 × 1,400 × 62 × 400 samples float32 is 2.8 GB). | EEG 11.95 GB per subject (HF file list), 141 GB for the full repo, videos 2.6 GB. |
| Evidence on signal | The EEG2Video paper reports decoding of coarse concept/meta information. An independent re-analysis ([Beyond Reconstruction, arXiv 2505.21385](https://arxiv.org/html/2505.21385v4)) found its EEG-only 40-way classification at **2.9% vs 2.5% chance**, i.e. coarse stimulus-level information, not trial-specific. | The paper reports no EEG-only benchmark; its EEG-only ablation of the full model is a reconstruction metric (50-way 0.30), not retrieval. |
| Expected value for us | Low-medium: the closest paradigm (natural video clips, own stimuli, channel names match), but 2-s silent(?) clips of 40 concepts, back-to-back presentation, and weak single-trial signal. | Low: audiovisual and large, but continuous, in-scanner, 30 Hz low-pass, approximate montage, copyrighted speech content. |

### 9.3 Considered and rejected

| Dataset | Why it does not suit |
|---|---|
| EEG-Video Action = [KU Leuven selective-attention dataset, Zenodo 15665101](https://zenodo.org/records/15665101) | Only **13 distinct YouTube videos** (via `yt-dlp` links; 64-ch BioSemi 2048 Hz, 19 subjects). Far too few targets for a ridge map, and the stimuli are link-only. |
| Visual touch EEG ([Sci Data 2026](https://www.nature.com/articles/s41597-026-06714-5), OpenNeuro ds005662, stimuli on [OSF jvkqa](https://osf.io/jvkqa/)) | 80 subjects, 64-ch BioSemi, 2,880 trials each, but **90 distinct 600-ms hand-touch clips**: a tiny, narrow stimulus set. |
| [Audiovisual long-term-memory EEG, ds006142](https://www.nature.com/articles/s41597-025-06203-1) | 660 ten-second film clips, 27 subjects, 64-ch; **videos cannot be released**; memory-retrieval task. |
| HBN-EEG ([bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2.full)) | A few continuous movies, no per-clip design. |
| FACED, DEAP, MAHNOB-HCI, AMIGOS, SEED-IV (emotion sets) | 16-40 long (1-4 min) emotion-elicitation clips; the labels are affect, not content. SEED-IV, MAHNOB, DEAP stimuli are not freely redistributable **(unverified)**. |
| EEG-SVRec ([arXiv 2404.01008](https://arxiv.org/abs/2404.01008)) | Short-video browsing on a recommendation platform, not controlled clip trials. |
| NKI EEG-fMRI naturalistic viewing ([INDI](https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html)) | Few movies, in-scanner EEG. |
| Image datasets (THINGS-EEG2, Alljoined, EEG-ImageNet, NOD) | §4 and §8; domain gap already measured. |

I found no other public EEG dataset with seconds-long, discretely presented, many-clip natural video released before 2026-10-16. The 2025-26 EEG-video papers I found (EEG-Mirror, MindCross, MindEV, MindCine, DynaMind, EEGVid) all use SEED-DV, EEG-Video Action or SEED.

### 9.4 Ranked shortlist and recommendation

1. **SEED-DV**: the only candidate that matches on the dimensions that matter (discrete natural video clips we can featurize ourselves, one view each, names matching our 17 channels). Its drawbacks are the application and licence, the 2-s clips, the grouped presentation, and weak published signal.
2. **CineBrain**: open and large, but continuous viewing, in-scanner EEG and an EGI montage make onset-locked zero-shot transfer unlikely.
3. Nothing else is suitable (§9.3). AVMoments-EEG would be the best on paper but is excluded as an MiT-family source (§9.1), and its stimuli are withheld.

**Recommendation (honest): no dataset here is likely to change the result by much.** The THINGS lesson (zero-shot about 2x chance, no ensemble gain) mostly came from a different stimulus type, and SEED-DV fixes only part of that: it is natural video but coarser, shorter and with a worse onset structure than our 3-s clips, and the independent re-analysis found its per-trial signal close to chance. If a further external-data attempt is wanted, the cheapest, lowest-risk path is:
(a) **ask the organizers whether an application-gated dataset counts as "public"**, and if so, have the user (an academic affiliation is needed) submit the SEED-DV application now, since approval time is unknown; it costs nothing while the other tracks run;
(b) only when access arrives, run the Step-0 test below. Do **not** download CineBrain unless SEED-DV fails to arrive and a second attempt is wanted.

### 9.5 Step-0 go/no-go sketches (same logic as §8.3; none can run until approved)

**SEED-DV (after access; ~2 h of work, no GPU beyond a minute):**
1. Embed the 1,400 clips with the same CLIP B-32 tower and the same frame sampling as ours (6 frames per 2-s clip; optionally V-JEPA 2 at 16 frames).
2. Load the 62-ch EEG, keep our 17 channels by name in our order, low-pass 40 Hz, resample 200 → 100 Hz, take 50-600 ms after each clip onset. Two variants: **all clips**, and **first clip of each 5-clip group only** (clean onset; about 280 per subject). Apply our noise-whitening and per-trial z-scoring (as §8.2).
3. Fit the whitened ridge decoder EEG → CLIP on 6-8 SEED-DV subjects (about 9-11k trials), alpha by held-out SEED-DV videos; sanity check within SEED-DV (retrieval among held-out clips).
4. Apply zero-shot to our EEG (dev sub-01..04, 102 candidates). **Kill if top-1 < ~2.5% on the 408 dev trials** (the §8.3 rule; noise of chance). If above, test as an ensemble member and as a ridge prior.
Data needed: probably 3-6 GB of EEG (unverified) plus the video blocks.

**CineBrain (only as a second attempt; ~3 h, download about 15 GB):** subject 1 `EEG_preprocessed_data.tar` (11.95 GB) + `videos.tar` (2.6 GB); map HydroCel-64 sites to our 17 by nearest 10-10 position (MNE montage `GSN-HydroCel-64_1.0`; some are substitutes); use the first 0.8-s segment of each 4-s clip as the response; CLIP of the first second of the clip; same ridge and zero-shot scoring and kill rule. Expected outcome: chance (no onset-locked response, approximate montage).

### 9.6 Sources (accessed 2026-10-18)
[SEED-DV project page](https://bcmi.sjtu.edu.cn/home/eeg2video/) · [EEG2Video code](https://github.com/XuanhaoLiu/EEG2Video) · [EEG2Video NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/84bad835faaf48f24d990072bb5b80ee-Paper-Conference.pdf) · [Beyond Reconstruction (EEGVid)](https://arxiv.org/html/2505.21385v4) · [SEED layout excerpt](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) ·
[CineBrain HF](https://huggingface.co/datasets/Fudan-fMRI/CineBrain) · [CineBrain paper](https://arxiv.org/html/2503.06940v1) · [CineBrain code](https://github.com/yanweifu-sii/CineBrain) · [AVMoments-EEG Zenodo](https://zenodo.org/records/22227356) · [AVMoments-EEG code](https://github.com/ShuningTangg/AVMoments-EEG) · [AVMIT](https://arxiv.org/html/2308.09685) ·
[KU Leuven EEG-video Zenodo](https://zenodo.org/records/15665101) · [visual touch EEG](https://www.nature.com/articles/s41597-026-06714-5) · [ds006142 memory-retrieval EEG](https://www.nature.com/articles/s41597-025-06203-1) · [EEG-SVRec](https://arxiv.org/abs/2404.01008) · [HBN-EEG](https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2.full) · [NKI naturalistic viewing](https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html).
