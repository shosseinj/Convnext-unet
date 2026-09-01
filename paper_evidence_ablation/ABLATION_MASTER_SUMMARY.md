# Ablation Master Summary

## Scope and evidence rule

This package audits the path to `one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine`. Values come from registry entries, runners, checkpoint metadata, training summaries, and evaluation summaries—not manuscript text. `--` means no verified value is available. All multi-seed means below use raw per-seed evaluation values before display rounding.

## Candidate classification

| ID | Architecture | Seeds / completion | Parameters | Status | Reason |
|---|---|---|---:|---|---|
| Exp01 | ConvNeXt-Tiny U-Net baseline | 42, 6543, 7777 complete | 29,190,098 | PROTOCOL MISMATCH | Old 350-epoch plateau/staged-unfreeze recipe differs materially from Exp45. |
| Exp03 | baseline + bottleneck FAFEM | 42, 6543, 7777 complete | 29,359,875 | PROTOCOL MISMATCH | Clean architecture isolation against Exp01, but same old training protocol. |
| Exp33 | bottleneck FAFEM only | 42, 6543, 7777 complete | 29,359,875 | VALID FOR FINAL ABLATION | Three seeds; same final-protocol recipe as Exp37/45. |
| Exp37 | Exp33 + Stage-3 MSCB-lite | 42, 6543, 7777 complete | 29,983,491 | VALID FOR FINAL ABLATION | Three seeds and one isolated architecture change from Exp33. |
| Exp41 | FAFEM + MSCB-lite at Stages 3 and 2 | seed 42 complete | 30,147,843 | EXPLORATORY ONLY | Protocol-compatible placement result, but only one seed and not the final residual-guidance mechanism. |
| Exp42 | FAFEM + MSCB-lite at Stages 3, 2 and 1 | seed 42 complete | 30,193,155 | EXPLORATORY ONLY | Protocol-compatible all-decoder-level placement result, but only one seed and unguided MSCB. |
| Exp43 | FAFEM + frequency-guided MSCB-lite at Stage 3 | seed 42 complete | 30,057,414 | VALID FOR FINAL ABLATION | Protocol-matched seed-42 bridge that directly introduces frequency guidance. |
| Exp44 | FAFEM + residual frequency-guided MSCB-lite at Stage 3, default init | seed 42 incomplete; no training/evaluation summary | -- | INCOMPLETE | Checkpoints/history exist, but no completion or evaluation evidence. |
| Exp45 | Exp44 mechanism + deterministic stronger initial guidance `0.05` | 42, 6543, 7777 complete | 30,057,415 | VALID FOR FINAL ABLATION | Final selected model; three verified seeds. |
| Exp48 | residual frequency-guided MSCB on all skips, batch 24 | no run directory | -- | NOT VERIFIED | Configuration exists but no run evidence. |
| Exp49 | same all-skips mechanism, batch 32 | seed 42 incomplete; no evaluation | -- | PROTOCOL MISMATCH | Batch differs from Exp45 and the run is incomplete. |

## Exact protocol/configuration

For Exp33/37/41/42/43/44/45/48: batch 24; 200 epochs; AdamW; decoder/new-module LR `3e-4`; encoder top-stage LR `3e-5`; encoder layerwise decay `0.8`; weight decay decoder `1e-4`, encoder `5e-2`, new layers `1e-2`; 5-epoch linear warmup then cosine to `1e-6`; no encoder freeze; uniform sampling; no frequency augmentation; no deep supervision; `DiceBCEBoundaryLoss(0.55 Dice + 0.25 BCE + 0.20 boundary, label smoothing 0.02)`; EMA `0.995`; AMP FP16; threshold `0.45`; no TTA; strict best merged-validation IoU checkpoint. Exp49 changes only batch to 32 at the protocol level. Exp01/03 use batch 24, default 350 epochs, standard optimizer grouping, decoder LR `4e-4`, encoder LR `4e-5`, plateau scheduler, decoder warmup 15, and plateau-driven staged encoder unfreezing; exact serialized threshold/TTA fields are unavailable.

## Dataset results

Values for three-seed experiments are means. Exp41/42/43 are seed 42. Columns: Dice, IoU, S-measure, weighted F, max E, mean E, MAE.

| Exp | Dataset | mDice | mIoU | S | Fw | maxE | meanE | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 33 | Kvasir-SEG | 0.925958 | 0.875879 | 0.931344 | 0.904809 | 0.963790 | 0.957005 | 0.024832 |
| 33 | CVC-ClinicDB | 0.941093 | 0.892336 | 0.950125 | 0.929094 | 0.988588 | 0.980794 | 0.010166 |
| 33 | CVC-300 | 0.886358 | 0.813114 | 0.930591 | 0.861206 | 0.971418 | 0.957284 | 0.007121 |
| 33 | CVC-ColonDB | 0.783536 | 0.703356 | 0.856964 | 0.761539 | 0.893737 | 0.887741 | 0.034361 |
| 33 | ETIS-LaribPolypDB | 0.781220 | 0.702697 | 0.872478 | 0.746788 | 0.905180 | 0.889078 | 0.014925 |
| 37 | Kvasir-SEG | 0.930440 | 0.881375 | 0.932583 | 0.910344 | 0.966499 | 0.959860 | 0.024487 |
| 37 | CVC-ClinicDB | 0.942365 | 0.893761 | 0.951526 | 0.931358 | 0.988041 | 0.981872 | 0.010019 |
| 37 | CVC-300 | 0.887153 | 0.815503 | 0.930695 | 0.863248 | 0.969692 | 0.959010 | 0.007061 |
| 37 | CVC-ColonDB | 0.790083 | 0.710781 | 0.859520 | 0.769133 | 0.899166 | 0.892844 | 0.035700 |
| 37 | ETIS-LaribPolypDB | 0.786395 | 0.707078 | 0.875170 | 0.750544 | 0.906818 | 0.890954 | 0.013626 |
| 41 | Kvasir-SEG | 0.928956 | 0.879109 | 0.931297 | 0.907866 | 0.965155 | 0.958473 | 0.024307 |
| 41 | CVC-ClinicDB | 0.938980 | 0.889857 | 0.949244 | 0.926764 | 0.983803 | 0.978810 | 0.010324 |
| 41 | CVC-300 | 0.884139 | 0.814161 | 0.932876 | 0.859968 | 0.969345 | 0.952405 | 0.007183 |
| 41 | CVC-ColonDB | 0.783703 | 0.702129 | 0.854491 | 0.759414 | 0.888478 | 0.882864 | 0.036048 |
| 41 | ETIS-LaribPolypDB | 0.784266 | 0.707254 | 0.876422 | 0.750364 | 0.914263 | 0.894522 | 0.014105 |
| 42 | Kvasir-SEG | 0.929456 | 0.879684 | 0.934002 | 0.908391 | 0.967732 | 0.959949 | 0.023914 |
| 42 | CVC-ClinicDB | 0.943985 | 0.895537 | 0.952415 | 0.934602 | 0.988437 | 0.983858 | 0.009862 |
| 42 | CVC-300 | 0.890467 | 0.818700 | 0.933556 | 0.865026 | 0.971646 | 0.960548 | 0.006866 |
| 42 | CVC-ColonDB | 0.788440 | 0.710419 | 0.857997 | 0.769071 | 0.898603 | 0.891932 | 0.034016 |
| 42 | ETIS-LaribPolypDB | 0.777452 | 0.702795 | 0.872355 | 0.745153 | 0.895557 | 0.884761 | 0.013612 |
| 43 | Kvasir-SEG | 0.929467 | 0.879563 | 0.931757 | 0.908768 | 0.966281 | 0.959629 | 0.024754 |
| 43 | CVC-ClinicDB | 0.937848 | 0.889767 | 0.947926 | 0.925352 | 0.983299 | 0.977671 | 0.011094 |
| 43 | CVC-300 | 0.868726 | 0.799952 | 0.921044 | 0.845123 | 0.957840 | 0.947695 | 0.007740 |
| 43 | CVC-ColonDB | 0.795254 | 0.711912 | 0.860085 | 0.769127 | 0.902561 | 0.896827 | 0.034685 |
| 43 | ETIS-LaribPolypDB | 0.786127 | 0.706470 | 0.874720 | 0.749244 | 0.921079 | 0.903939 | 0.014971 |
| 45 | Kvasir-SEG | 0.931284 | 0.882594 | 0.933068 | 0.911316 | 0.966485 | 0.960168 | 0.023960 |
| 45 | CVC-ClinicDB | 0.937031 | 0.887735 | 0.946646 | 0.925386 | 0.982493 | 0.977577 | 0.010868 |
| 45 | CVC-300 | 0.882061 | 0.810394 | 0.928071 | 0.857084 | 0.970596 | 0.955810 | 0.007368 |
| 45 | CVC-ColonDB | 0.794351 | 0.714178 | 0.861236 | 0.771516 | 0.905043 | 0.898314 | 0.034861 |
| 45 | ETIS-LaribPolypDB | 0.786980 | 0.707118 | 0.877966 | 0.753165 | 0.915691 | 0.902404 | 0.014617 |

Old Exp01/03 measurements are retained machine-readably in `ABLATION_RESULTS.csv`; they are not repeated as paper candidates because of the protocol mismatch.

## Cleanest defensible chain

The closest defensible chain is **Exp33 FAFEM -> Exp37 FAFEM + Stage-3 MSCB-lite -> Exp43 frequency-guided Stage-3 MSCB-lite -> Exp45 residual frequency-guided Stage-3 MSCB-lite with stronger initialization**. Exp33 and Exp37 have three seeds. Exp43 has only seed 42, so comparisons through Exp43/45 must use paired seed 42, not a three-seed mean versus a single seed. Exp44 is incomplete; therefore residual guidance and stronger initialization cannot be separated with completed evidence. There is no protocol-compatible no-FAFEM baseline in this final recipe.

## Valid paired seed-42 deltas

Percentage points equal raw absolute delta multiplied by 100.

| Comparison | Dataset | Delta mDice | pp | Delta mIoU | pp | Delta MAE | pp |
|---|---|---:|---:|---:|---:|---:|---:|
| Exp33 -> Exp37 | Kvasir-SEG | +0.002963 | +0.30 | +0.003594 | +0.36 | +0.000273 | +0.03 |
| Exp33 -> Exp37 | CVC-ClinicDB | +0.000265 | +0.03 | -0.001563 | -0.16 | +0.000058 | +0.01 |
| Exp33 -> Exp37 | CVC-300 | +0.007824 | +0.78 | +0.011356 | +1.14 | -0.000772 | -0.08 |
| Exp33 -> Exp37 | CVC-ColonDB | +0.009258 | +0.93 | +0.010582 | +1.06 | +0.000607 | +0.06 |
| Exp33 -> Exp37 | ETIS | +0.013049 | +1.30 | +0.009326 | +0.93 | -0.002407 | -0.24 |
| Exp37 -> Exp43 | Kvasir-SEG | -0.001089 | -0.11 | -0.001861 | -0.19 | -0.000041 | -0.00 |
| Exp37 -> Exp43 | CVC-ClinicDB | -0.001311 | -0.13 | +0.001421 | +0.14 | +0.000574 | +0.06 |
| Exp37 -> Exp43 | CVC-300 | -0.021358 | -2.14 | -0.021110 | -2.11 | +0.000873 | +0.09 |
| Exp37 -> Exp43 | CVC-ColonDB | +0.000330 | +0.03 | -0.002656 | -0.27 | -0.000805 | -0.08 |
| Exp37 -> Exp43 | ETIS | -0.001804 | -0.18 | -0.001655 | -0.17 | +0.002077 | +0.21 |
| Exp43 -> Exp45 | Kvasir-SEG | +0.004102 | +0.41 | +0.006396 | +0.64 | -0.001413 | -0.14 |
| Exp43 -> Exp45 | CVC-ClinicDB | -0.003605 | -0.36 | -0.006362 | -0.64 | +0.000036 | +0.00 |
| Exp43 -> Exp45 | CVC-300 | +0.014367 | +1.44 | +0.012380 | +1.24 | -0.000073 | -0.01 |
| Exp43 -> Exp45 | CVC-ColonDB | +0.008196 | +0.82 | +0.010433 | +1.04 | -0.000970 | -0.10 |
| Exp43 -> Exp45 | ETIS | +0.011844 | +1.18 | +0.012726 | +1.27 | -0.000167 | -0.02 |

Negative MAE is improvement; positive MAE is degradation. Deltas above use seed 42 raw JSON values. Exp45 three-seed results are referenced from `paper_evidence_exp45/EXP45_THREE_SEED_RESULTS.md` and snapshotted without recalculation conflicts.

## Placement ablation

The measured placement sequence is unguided MSCB-lite: Stage 3 only (Exp37 seed 42), Stages 3+2 (Exp41), and Stages 3+2+1 (Exp42). It is protocol-clean at seed 42, but does not test residual frequency-guided placement. The configured residual all-skips variants Exp48/49 have no completed evaluation. No untested placement is reported.

# Recommended Final Ablation Tables

## Table A: Core architectural progression

- Include paired seed-42 Exp33, Exp37, Exp43, and Exp45.
- Report all five datasets with mDice, mIoU, and MAE; optionally S-measure for space permitting.
- This is protocol-matched and follows actual architecture flags, but the caption must disclose that Exp43 is single-seed and that Exp43 -> Exp45 combines residualization with stronger initialization.

## Table B: MSCB placement

- Include seed-42 Exp37, Exp41, and Exp42.
- Report mDice/mIoU/MAE on all five datasets.
- This cleanly tests Stage 3 versus Stages 3+2 versus Stages 3+2+1 for unguided MSCB-lite. Do not label it as residual-guidance placement.

Do not put Exp01/03, Exp44, Exp48, or Exp49 in a final quantitative ablation table. Exp45's three-seed robustness should remain a separate final-model table linked to the existing Exp45 evidence package.
