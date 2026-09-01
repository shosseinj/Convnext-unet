# Final Experiment Identification

- **Experiment:** `one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine`
- **Completed seeds:** 42, 6543, 7777.
- **Architecture:** ConvNeXt-Tiny U-Net with bottleneck FAFEM and residual frequency-guided MSCB-lite at the Stage-3 decoder/skip fusion. The three final checkpoints have identical recorded architecture metadata.
- **Input:** 352 x 352 RGB.
- **Prediction threshold:** 0.45.
- **TTA:** disabled for all three final evaluations.
- **Checkpoint selection:** strictly improved validation IoU (`test_acc = test_iou`) on the merged held-out Kvasir-SEG and CVC-ClinicDB validation split, evaluated with EMA weights.
- **Best checkpoint epochs:** seed 42 = 184; seed 6543 = 175; seed 7777 = 137.
- **Aggregation:** arithmetic mean and sample standard deviation across the three seeds (`ddof=1`). Calculations use the unrounded values from each `evaluation_summary.json`; displayed values are rounded to four decimal places.

## CVC-300

| Metric | Seed 42 | Seed 6543 | Seed 7777 | Mean | Sample Std | Paper value |
|---|---:|---:|---:|---:|---:|---:|
| mDice | 0.8831 | 0.8803 | 0.8828 | 0.8821 | 0.0016 | 0.8821 ± 0.0016 |
| mIoU | 0.8123 | 0.8089 | 0.8099 | 0.8104 | 0.0018 | 0.8104 ± 0.0018 |
| S-measure | 0.9296 | 0.9246 | 0.9300 | 0.9281 | 0.0030 | 0.9281 ± 0.0030 |
| Weighted F-measure | 0.8582 | 0.8567 | 0.8564 | 0.8571 | 0.0010 | 0.8571 ± 0.0010 |
| Max E-measure | 0.9701 | 0.9694 | 0.9723 | 0.9706 | 0.0015 | 0.9706 ± 0.0015 |
| Mean E-measure | 0.9526 | 0.9612 | 0.9535 | 0.9558 | 0.0047 | 0.9558 ± 0.0047 |
| MAE | 0.0077 | 0.0072 | 0.0072 | 0.0074 | 0.0003 | 0.0074 ± 0.0003 |

## CVC-ClinicDB

| Metric | Seed 42 | Seed 6543 | Seed 7777 | Mean | Sample Std | Paper value |
|---|---:|---:|---:|---:|---:|---:|
| mDice | 0.9342 | 0.9386 | 0.9382 | 0.9370 | 0.0024 | 0.9370 ± 0.0024 |
| mIoU | 0.8834 | 0.8903 | 0.8895 | 0.8877 | 0.0038 | 0.8877 ± 0.0038 |
| S-measure | 0.9453 | 0.9473 | 0.9473 | 0.9466 | 0.0012 | 0.9466 ± 0.0012 |
| Weighted F-measure | 0.9223 | 0.9265 | 0.9273 | 0.9254 | 0.0027 | 0.9254 ± 0.0027 |
| Max E-measure | 0.9817 | 0.9831 | 0.9826 | 0.9825 | 0.0007 | 0.9825 ± 0.0007 |
| Mean E-measure | 0.9768 | 0.9781 | 0.9779 | 0.9776 | 0.0007 | 0.9776 ± 0.0007 |
| MAE | 0.0111 | 0.0106 | 0.0109 | 0.0109 | 0.0003 | 0.0109 ± 0.0003 |

## CVC-ColonDB

| Metric | Seed 42 | Seed 6543 | Seed 7777 | Mean | Sample Std | Paper value |
|---|---:|---:|---:|---:|---:|---:|
| mDice | 0.8034 | 0.7857 | 0.7939 | 0.7944 | 0.0089 | 0.7944 ± 0.0089 |
| mIoU | 0.7223 | 0.7052 | 0.7150 | 0.7142 | 0.0086 | 0.7142 ± 0.0086 |
| S-measure | 0.8666 | 0.8548 | 0.8624 | 0.8612 | 0.0060 | 0.8612 ± 0.0060 |
| Weighted F-measure | 0.7778 | 0.7647 | 0.7720 | 0.7715 | 0.0066 | 0.7715 ± 0.0066 |
| Max E-measure | 0.9092 | 0.9019 | 0.9040 | 0.9050 | 0.0038 | 0.9050 ± 0.0038 |
| Mean E-measure | 0.9029 | 0.8953 | 0.8967 | 0.8983 | 0.0040 | 0.8983 ± 0.0040 |
| MAE | 0.0337 | 0.0358 | 0.0351 | 0.0349 | 0.0010 | 0.0349 ± 0.0010 |

## ETIS-LaribPolypDB

| Metric | Seed 42 | Seed 6543 | Seed 7777 | Mean | Sample Std | Paper value |
|---|---:|---:|---:|---:|---:|---:|
| mDice | 0.7980 | 0.7877 | 0.7753 | 0.7870 | 0.0113 | 0.7870 ± 0.0113 |
| mIoU | 0.7192 | 0.7070 | 0.6951 | 0.7071 | 0.0120 | 0.7071 ± 0.0120 |
| S-measure | 0.8821 | 0.8766 | 0.8752 | 0.8780 | 0.0037 | 0.8780 ± 0.0037 |
| Weighted F-measure | 0.7617 | 0.7499 | 0.7479 | 0.7532 | 0.0074 | 0.7532 ± 0.0074 |
| Max E-measure | 0.9176 | 0.9170 | 0.9124 | 0.9157 | 0.0028 | 0.9157 ± 0.0028 |
| Mean E-measure | 0.9048 | 0.9024 | 0.9000 | 0.9024 | 0.0024 | 0.9024 ± 0.0024 |
| MAE | 0.0148 | 0.0153 | 0.0138 | 0.0146 | 0.0008 | 0.0146 ± 0.0008 |

## Kvasir-SEG

| Metric | Seed 42 | Seed 6543 | Seed 7777 | Mean | Sample Std | Paper value |
|---|---:|---:|---:|---:|---:|---:|
| mDice | 0.9336 | 0.9262 | 0.9341 | 0.9313 | 0.0044 | 0.9313 ± 0.0044 |
| mIoU | 0.8860 | 0.8755 | 0.8864 | 0.8826 | 0.0062 | 0.8826 ± 0.0062 |
| S-measure | 0.9345 | 0.9301 | 0.9346 | 0.9331 | 0.0026 | 0.9331 ± 0.0026 |
| Weighted F-measure | 0.9145 | 0.9052 | 0.9143 | 0.9113 | 0.0053 | 0.9113 ± 0.0053 |
| Max E-measure | 0.9683 | 0.9634 | 0.9678 | 0.9665 | 0.0026 | 0.9665 ± 0.0026 |
| Mean E-measure | 0.9620 | 0.9571 | 0.9615 | 0.9602 | 0.0027 | 0.9602 ± 0.0027 |
| MAE | 0.0233 | 0.0248 | 0.0238 | 0.0240 | 0.0007 | 0.0240 ± 0.0007 |

## Model Complexity

- **Trainable parameters:** 30,057,415.
- **Total parameters:** 30,057,415.
- **Complexity:** 27.818488992 GFLOPs at 1 x 3 x 352 x 352.
- **FLOP convention:** THOP counts 13.909244496 GMACs; the project reports FLOPs as `2 × MACs`.

## Paper-ready summary

Across three seeds, Kvasir-SEG mDice = 0.9313 ± 0.0044 and mIoU = 0.8826 ± 0.0062. CVC-ClinicDB mDice = 0.9370 ± 0.0024 and mIoU = 0.8877 ± 0.0038. CVC-300 mDice = 0.8821 ± 0.0016 and mIoU = 0.8104 ± 0.0018. CVC-ColonDB mDice = 0.7944 ± 0.0089 and mIoU = 0.7142 ± 0.0086. ETIS-LaribPolypDB mDice = 0.7870 ± 0.0113 and mIoU = 0.7071 ± 0.0120.
