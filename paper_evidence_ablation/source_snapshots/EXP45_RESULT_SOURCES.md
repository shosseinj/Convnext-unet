# Exp.45 Result Sources and Verification

## Per-seed evidence

| Seed | Final checkpoint | Evaluation artifact | Best epoch | Completion |
|---:|---|---|---:|---|
| 42 | `one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_42/best_checkpoint.pth` | `.../seed_42/evaluation_summary.json` | 184 | `max_epochs`; final epoch 199 |
| 6543 | `one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_6543/best_checkpoint.pth` | `.../seed_6543/evaluation_summary.json` | 175 | `max_epochs`; final epoch 199 |
| 7777 | `one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_7777/best_checkpoint.pth` | `.../seed_7777/evaluation_summary.json` | 137 | `early_stopping`; final epoch 167 |

All three `evaluation_summary.json` files identify the same Exp.45 name, their matching seed, threshold 0.45, `tta=false`, total/trainable parameters 30,057,415, and 27.818488992 GFLOPs. The recorded checkpoint architecture dictionaries are identical across the three seeds.

## Implementation sources

| Reported item | Authoritative source |
|---|---|
| Final per-seed metrics, parameter counts, MACs/FLOPs, threshold, TTA | `seed_<seed>/evaluation_summary.json` |
| Checkpoint identity, architecture metadata, best epoch, selection metric | `seed_<seed>/best_checkpoint.pth` |
| Completion status and final epoch | `seed_<seed>/training_summary.json` |
| Metric implementation and per-image aggregation | `evaluation_core.evaluate_loader`, `evaluation_core.binary_metrics_per_image` |
| Checkpoint loading/validation | `evaluate.py` |
| Parameter counting and THOP complexity conversion | `evaluation_core.count_parameters`, `evaluation_core.measure_complexity` |
| Checkpoint selection rule | `main_torch.py` (`test_acc = test_iou`; strict improvement) |

## Metric names

- `mDice`, `mIoU`: mean of per-image thresholded binary Dice and IoU.
- `S_alpha`: S-measure.
- `F_beta_w`: weighted F-measure.
- `maxE_phi`, `mE_phi`: maximum and mean E-measure respectively.
- `MAE`: mean absolute error.

## Conflicts and NOT VERIFIED

- No conflict was found among the three final evaluation artifacts, checkpoint identities, architecture metadata, threshold, TTA state, parameter count, or GFLOPs.
- The three runs have different completion modes (two reached maximum epochs and seed 7777 used early stopping). This is documented provenance, not a metric conflict.
- Hardware/GPU model is **NOT VERIFIED** from the final result artifacts.
