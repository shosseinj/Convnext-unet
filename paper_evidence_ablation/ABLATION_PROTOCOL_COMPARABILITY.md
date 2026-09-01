# Ablation Protocol Comparability

The final-model protocol group (`P-WC200-LW`) is defined by the actual Exp33/37/41/42/43/44/45/48 runners: batch 24, 200 epochs, AdamW, decoder LR `3e-4`, encoder top-stage LR `3e-5`, layer decay `0.8`, warmup-cosine with 5 warmup epochs and `1e-6` minimum LR, no encoder freeze, threshold `0.45`, best checkpoint by strict improvement in merged validation IoU, EMA (`0.995`) scoring/saving, AMP FP16, and no evaluation TTA. Exp49 changes batch size and is a separate protocol group.

| Experiment | Batch | Epochs | Scheduler | LR policy | Threshold | Checkpoint rule | TTA | Comparable to Exp45? |
|---|---:|---:|---|---|---:|---|---|---|
| Old baseline (Exp01) | 24 | 350 | ReduceLROnPlateau | standard AdamW; decoder `4e-4`, encoder `4e-5`; staged plateau unfreeze | NOT AVAILABLE in result JSON | validation IoU; exact historical evaluator fields absent | NOT AVAILABLE | No |
| Old FAFEM (Exp03) | 24 | 350 | ReduceLROnPlateau | same old baseline policy | NOT AVAILABLE in result JSON | validation IoU; exact historical evaluator fields absent | NOT AVAILABLE | No |
| Exp33 FAFEM | 24 | 200 | 5-epoch warmup + cosine | layerwise AdamW; `3e-4`/`3e-5`; decay `0.8` | 0.45 by evaluator source; omitted from older JSON schema | strict merged-validation IoU | false by invocation; omitted from older JSON schema | Yes |
| Exp37 FAFEM + Stage-3 MSCB-lite | 24 | 200 | same | same | 0.45 by evaluator source; omitted from older JSON schema | same | false by invocation; omitted from older JSON schema | Yes |
| Exp41 Stage-3+2 MSCB-lite | 24 | 200 | same | same | 0.45 | same | false | Yes, seed 42 only |
| Exp42 Stage-3+2+1 MSCB-lite | 24 | 200 | same | same | 0.45 | same | false | Yes, seed 42 only |
| Exp43 frequency-guided Stage-3 | 24 | 200 | same | same | 0.45 | same | false | Yes, seed 42 only |
| Exp44 residual frequency-guided Stage-3 | 24 | 200 | same | same | -- | same configured | -- | Config comparable; run incomplete |
| Exp45 residual frequency-guided Stage-3, stronger init | 24 | 200 | same | same | 0.45 | same | false | Reference |
| Exp48 residual frequency-guided all skips | 24 | 200 | same | same | -- | same configured | -- | Config comparable; no run artifacts |
| Exp49 all skips, batch 32 | 32 | 200 | same | same | -- | same configured | -- | No; batch mismatch and incomplete |

## Shared final-protocol details

- Data: the same repository training split/loader for Kvasir-SEG plus CVC-ClinicDB; external datasets are evaluation-only.
- Loss: `DiceBCEBoundaryLoss(dice_w=0.55, bce_w=0.25, boundary_w=0.20, label_smoothing=0.02)`. There is no deep supervision in these candidates. The UGBR composite loss requested by a different workflow is not used here.
- Weight decay: decoder `1e-4`, encoder `5e-2`, newly added/refinement layers `1e-2`.
- EMA/AMP: EMA enabled at decay `0.995`; AMP FP16 required by registry metadata.
- Input: `352 x 352 x 3` in the evaluation/complexity path and project protocol.
- Early stopping: patience 30. Exp45 seeds 42/6543 reached epoch 199; seed 7777 stopped at epoch 167. This is the same policy, not a protocol change.

## Material mismatches

The old baseline and old FAFEM runs differ from Exp45 in epoch budget, scheduler, LR profile, encoder-unfreeze policy, and weight-decay grouping. They are useful historical context but not clean controls for the final progression. Exp49 changes batch size from 24 to 32 and never completed. Older evaluation JSONs for Exp01/03/33/37 omit explicit `threshold` and `tta` fields; their runner/evaluator invocation supports 0.45/no-TTA, but the missing serialized fields are retained as a provenance limitation rather than silently backfilled.
