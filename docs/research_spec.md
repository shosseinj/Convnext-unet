# Reproducible Research Specification

## Objective

Evaluate BSEI-ConvNeXt-UNet polyp-segmentation variants under one immutable
data, training, checkpoint-selection, evaluation, and reporting protocol.
`BSEI` is the only official module name. `SimpleFusion` is normal skip fusion.

## Data contract

- Kvasir-SEG and CVC-ClinicDB are training/validation sources.
- Split manifests are generated once per seed and stored with filenames.
- Seeds are 42, 3407, and 2026.
- CVC-300, CVC-ColonDB, and ETIS-LARIBPOLYPDB are final test-only datasets.
- Test datasets cannot select a checkpoint, threshold, epoch, or configuration.
- Image/mask pairing, duplicates, unreadable files, and split overlap are checked.

## Shared training protocol

- Input: 352 by 352 RGB.
- ImageNet-pretrained ConvNeXt encoder for every applicable variant.
- AdamW, maximum 150 epochs.
- Freeze encoder in epochs 1-10; unfreeze at epoch 11.
- Encoder LR 1e-5; decoder and added-module LR 1e-4.
- Weight decay 1e-4 for all parameter groups.
- Early stopping patience 30 after full-network training begins.
- Identical loss, augmentation, scheduler, batch size, and threshold policy.
- TTA disabled for ablation selection and comparison.

## Selection and evaluation

The selection score is the equal dataset-level mean:

`selection_dice = (kvasir_val_dice + clinicdb_val_dice) / 2`

The chosen checkpoint is frozen before threshold selection and external testing.
Report Dice, IoU, precision, recall, specificity, pixel accuracy, and MAE per
image and per dataset. Empty-mask behavior is explicit and tested.

For each variant, aggregate each seed first and report the three-seed mean and
sample standard deviation. Polyp size bins use mask-area/image-area: small below
5%, medium 5%-20%, and large above 20%.

## Evidence contract

Every run stores resolved config, git commit, seed, split-manifest hashes,
environment versions, parameter count, FLOPs/MACs method, logs, best checkpoint,
and raw metrics. Tables and manuscript claims are generated only from validated
raw/aggregated JSON or CSV artifacts.

## Gates

1. All architectures instantiate, return correct shapes, report parameters and
   FLOPs, and reject incompatible checkpoints clearly.
2. Forward/backward, finite loss, save/load, and short epoch smoke tests pass.
3. All variants run a seed-42 pilot and receive review.
4. Approved variants run all three seeds with complete artifacts.
5. Statistical aggregation and size-bin analysis pass schema checks.
6. Tables, figures, and manuscript contain no unsupported values or placeholders.
