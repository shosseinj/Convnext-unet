# ConvNeXtUNet Ablation Training Protocol

This file is the protocol reminder for all ConvNeXtUNet polyp segmentation ablation runs. Every ablation variant must be trained with the same fair setup unless an experiment explicitly states otherwise.

## Ablation Variants

1. Baseline
2. MSC
3. MSC + BSEI
4. MSC + BSEI + DB
5. MSC + BSEI + DB + GDF
6. Full model

## Abbreviations

- MSC: MultiScaleContext
- BSEI: official module name; no expanded form is used in this project
- DB: DetailBranch
- GDF: GatedDetailFusion
- DS: Deep Supervision

## Required Shared Training Protocol

Use this exact training setup for every ablation variant:

- Input size: 352 x 352 x 3
- Training datasets: Kvasir-SEG and CVC-ClinicDB
- Encoder initialization: pretrained ConvNeXt-Tiny ImageNet weights for every variant
- Maximum epochs: 150
- Encoder freeze schedule:
  - Epochs 1-10: freeze encoder and train decoder / added modules
  - Epochs 11-150: unfreeze encoder and train the full network end-to-end
- Learning rates:
  - Encoder learning rate after unfreezing: 1e-5
  - Decoder / added module learning rate: 1e-4
- Weight decay: 1e-4
- Early stopping patience: 25-30 epochs, if early stopping is implemented
- Optimizer, scheduler, loss function, augmentation, thresholding rule, batch size, and random seed policy must be identical for all variants.
- If multiple seeds are used, use the same seeds for every variant.

## Validation And Checkpoint Selection

Save the best checkpoint by average validation Dice on Kvasir-SEG and CVC-ClinicDB:

```text
best_score = (kvasir_val_dice + clinicdb_val_dice) / 2
```

Do not select the best checkpoint using CVC-300, CVC-ColonDB, or ETIS-LARIBPOLYPDB. These are unseen test datasets and must be kept strictly for final cross-dataset testing.

## Final Cross-Dataset Testing

Use CVC-300, CVC-ColonDB, and ETIS-LARIBPOLYPDB only after checkpoint selection is complete. The selected checkpoint must come from Kvasir-SEG and CVC-ClinicDB validation Dice, not from these unseen test datasets.

## Pre-Run Checklist

Before each ablation training run, verify:

- [ ] The variant architecture matches the intended ablation row.
- [ ] Input size is 352 x 352 x 3.
- [ ] Training data uses Kvasir-SEG and CVC-ClinicDB.
- [ ] ConvNeXt-Tiny ImageNet encoder weights are loaded.
- [ ] Maximum epochs is 150.
- [ ] Encoder is frozen for epochs 1-10.
- [ ] Encoder is unfrozen from epoch 11.
- [ ] Encoder LR after unfreezing is 1e-5.
- [ ] Decoder / added module LR is 1e-4.
- [ ] Weight decay is 1e-4.
- [ ] Early stopping patience is 25-30 epochs, if enabled.
- [ ] Batch size matches the other ablation variants.
- [ ] Optimizer, scheduler, loss, augmentation, thresholding, and seed policy match the other variants.
- [ ] Best checkpoint is selected by average Kvasir-SEG and CVC-ClinicDB validation Dice.
- [ ] CVC-300, CVC-ColonDB, and ETIS-LARIBPOLYPDB are reserved only for final cross-dataset testing.
