# Evaluation Protocol

## Checkpoint and data procedure

`evaluate.py` loads only `best_checkpoint.pth`, rejects incomplete checkpoints, validates experiment/seed/architecture metadata, reconstructs the registered model, and loads the model state strictly. The seed-42 Exp.45 checkpoint fingerprint is `b979d16f4eff133af95f6d4cc0a4f86ce838837bbb99bf07f69a8154a4ae714b`.

Five datasets are evaluated sequentially: Kvasir-SEG held-out 10%, CVC-ClinicDB held-out 10%, then full CVC-300, CVC-ColonDB, and ETIS-LaribPolypDB. External results never select a checkpoint.

## Prediction and aggregation

- Model emits logits; `evaluation_core._main_logits` selects main logits.
- Sigmoid is applied once. Probabilities are bilinearly resized to mask size with `align_corners=False` only if shapes differ.
- Binary prediction is `probability > 0.45`.
- TTA is disabled. Although a five-view flip/rotation TTA function exists, final evaluation has `tta=false` and the runner does not pass `--tta`.
- Dice and IoU are computed **per image** from flattened binary masks using smoothing `1e-6`, then averaged across images (`mDice`, `mIoU`).
- S-measure (`S_alpha`), weighted F-measure (`F_beta_w`), E-measure curve mean/max (`mE_phi`, `maxE_phi`), and MAE use `py_sod_metrics`, stepped per image with 8-bit probability and binary ground truth.

Definitions in `evaluation_core.py`:

```text
Dice = (2 |P ∩ G| + 1e-6) / (|P| + |G| + 1e-6)
IoU  = (|P ∩ G| + 1e-6) / (|P| + |G| - |P ∩ G| + 1e-6)
```

Training validation uses `main_torch.py:test_segmentation`, whereas final five-dataset evaluation uses `evaluation_core.evaluate_loader`; they are not the same code path. Both use sigmoid and binary threshold 0.45 in Exp.45, but their metric aggregation/metric suite differs. Parameter count is `sum(p.numel())` over the reconstructed model; complexity is measured with THOP on a `1x3x352x352` dummy input.
