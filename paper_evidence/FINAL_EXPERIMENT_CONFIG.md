# Final Experiment Configuration

## Scope and source priority

This package documents only `one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine`, seed 42. The final checkpoint metadata was checked first, then the registered Exp.45 configuration, runner, training log, and receiving code. The best checkpoint is epoch 184; training ran through epoch 199/200 and completed by maximum epochs.

| Setting | Final value | Evidence source | Origin |
|---|---|---|---|
| Experiment | `one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine` | checkpoint `experiment_name`; registry; runner | CHECKPOINT/CONFIG/RUNNER |
| Seed | 42 | checkpoint `seed`; runner `-Seed` | CHECKPOINT/RUNNER |
| Input size | 352 x 352 RGB | `main_torch.py` `--input_size`; `evaluation_core.py:read_dataset` | CODE DEFAULT |
| Epochs | 200 requested; best epoch 184; final epoch 199 (zero-based) | runner `-Epochs 200`; checkpoint | RUNNER/CHECKPOINT |
| Batch size | 24 | runner `-BatchSize`; dry-run command | RUNNER |
| Optimizer | AdamW, betas `(0.9, 0.999)` | `main_torch.py` optimizer construction | CODE |
| Decoder LR | 3.0e-4 | checkpoint `optimizer_max_lrs.decoder`; log | CHECKPOINT/LOG |
| Refine LR | 4.5e-4 | checkpoint `optimizer_max_lrs.refine`; log | CHECKPOINT/LOG |
| Encoder stage 4/3/2/1+stem LR | 3.0e-5 / 2.4e-5 / 1.92e-5 / 1.536e-5 | checkpoint `optimizer_max_lrs`; log | CHECKPOINT/LOG |
| Layer-wise decay | 0.8 | checkpoint `encoder_layer_decay`; runner | CHECKPOINT/RUNNER |
| Weight decay: decoder / encoder / refine | 1.0e-4 / 5.0e-2 / 1.0e-2 | runner; optimizer log | RUNNER/LOG |
| Scheduler | five-epoch linear warmup (0.1 to 1.0 LR factor), then cosine annealing to 1.0e-6 | runner; `create_warmup_cosine_scheduler`; checkpoint `scheduler_type` | RUNNER/CODE/CHECKPOINT |
| Gradient clipping | global max norm 1.0 | runner `-MaxGradNorm 1.0` | RUNNER |
| Pretraining | `convnext_tiny_22k_1k_384.pth`; logged as ConvNeXt-Tiny ImageNet encoder weights | runner; training log | RUNNER/LOG |
| Encoder freeze policy | no staged freeze; all parameters trainable from epoch 1 | runner `-UnfreezeSchedule none`, `-DecoderWarmupEpochs 0`; code | RUNNER/CODE |
| EMA | enabled, decay 0.995; EMA weights used for validation/checkpoint scoring | log; `ModelEMA`; `scored_model_state_dict` | LOG/CODE |
| Precision | AMP FP16 | checkpoint `training_precision=amp_fp16`; runner `--amp True` | CHECKPOINT/RUNNER |
| Deep supervision | disabled (`0` heads) | checkpoint architecture; runner | CHECKPOINT/RUNNER |
| Frequency augmentation | disabled | checkpoint and training log | CHECKPOINT/LOG |
| Sampling | uniform | checkpoint `sampling_mode` | CHECKPOINT |
| Early stopping | configured patience 30, start epoch 0; not triggered (`completion_reason=max_epochs`) | runner default via Invoke; checkpoint | RUNNER/CHECKPOINT |
| Binary threshold | 0.45 | `evaluate.py`; final `evaluation_summary.json` | EVALUATION |
| TTA | disabled | runner `--tta_check False`; evaluation summary `tta=false` | RUNNER/EVALUATION |
| Hardware | NOT VERIFIED | No explicit GPU model was found in final checkpoint/log evidence | — |
| Framework | PyTorch 2.5.1+cu124 | final checkpoint metadata extraction | CHECKPOINT ENVIRONMENT |

## Loss

`DiceBCEBoundaryLoss` is selected for `one_seed_*` experiments: `0.55 * DiceLoss + 0.25 * BCEWithLogitsLoss(label-smoothed targets, smoothing=0.02) + 0.20 * BoundaryAwareLoss(kappa=5)`. Focal-Tversky is present in the class but has weight `0.0` in Exp.45. Source: `main_torch.py:DiceBCEBoundaryLoss` and its Exp.45 criterion selection.

## Checkpoint selection

At each epoch, validation is the merged held-out Kvasir-SEG and CVC-ClinicDB split. The EMA model is scored using threshold 0.45. `test_acc` is assigned to validation IoU, and a checkpoint is saved only when IoU is strictly greater than prior best. The final selected checkpoint reports validation Dice 0.9348011783 and IoU 0.8780511022 at epoch 184.

## Conflict and verification notes

- The checkpoint field `encoder_stages_unfrozen=0` is a legacy progress field. It conflicts semantically with the actual Exp.45 policy `unfreeze_schedule=none` and the code that makes all parameters trainable before optimizer construction. The runner/code policy is the correct description: no freezing.
- `cosine_t0=8` and `cosine_t_mult=2` are checkpoint fields, but are unused by the selected monotonic `warmup_cosine` scheduler; they are not part of the final scheduler configuration.
- The runner defaults early stopping to 30/0 through `Invoke-OneSeed-Ablation.ps1`; it did not terminate this completed run.
