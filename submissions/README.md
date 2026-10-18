# Submission sets (sub-05 / sub-06)

| Directory | Model | Dev (sub-01..04) top-1 / top-5 / median | Status |
|---|---|---|---|
| `final_csls/` | **Headline.** Whitened decoders on CLIP B-32 (w 0.0), DINOv2-B patch (0.4), V-JEPA 2-L 16 f (0.3), CLIP block-8 CLS (0.1) + contrastive two-tower on CLIP (0.2, 3 seeds); test-set centering; CSLS k=5 per member (allowed by the organizers) | 24.0 / 57.1 / 4.25 | final |
| `final_nocsls/` | Backup without CSLS: the same members + Stage 3 EEGNet, no centering | 19.1 / 50.0 / 5.5 | backup |
| `decw_clip_b32_f8_t1.5s_w5-60/` | Single whitened CLIP decoder (M8a) | 14.7 / 42.2 / 7.5 | simple fallback |

Weights chosen by training-only CV (leave-one-fold-out strategy rule). Dev labels were used only to report.
Reproduce the headline (branch track-matching, commit 7297a59):
`PYTHONPATH=src:scripts python scripts/ensemble.py --member decw:clip_b32 --member decw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 --member con:clip_b32:two_tower_d256 --member decw:clip_b32_mid8_all:8 --max-seconds 1.5 --center-test --csls 5 --csls-stage member --experiment fin5_ctr_m5`
