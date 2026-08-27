# Current Polyp Segmentation Experimental Setup

This file records the exact configuration currently implemented in the codebase for the live one-seed ConvNeXt U-Net run.

## Current live experiment

- Experiment: `one_seed_32_fafem_layerwise_warmup_cosine`
- Seed: `42`
- Output directory: `one_seed_results/ablation/32_fafem_layerwise_warmup_cosine/seed_42/`

## Model

- Backbone/encoder: ConvNeXt-Tiny
- Pretrained weights: `convnext_tiny_22k_1k_384.pth`
- Decoder architecture: ConvNeXt U-Net decoder with `decoder4`, `decoder3`, `decoder2`, `decoder1`, skip fusion blocks `bsei4`, `bsei3`, `bsei2`, `bsei1`, and `final_refine`
- Input resolution: `352 x 352`
- Total parameters: `29,359,875`
- Enabled modules: bottleneck FAFEM only
- Module placement: FAFEM is applied to bottleneck feature `f4` before the bottleneck block and decoder

## Dataset

- Training datasets: Kvasir-SEG and CVC-ClinicDB
- Validation datasets: held-out 10% split from each of Kvasir-SEG and CVC-ClinicDB
- Split: 90% train / 10% validation per dataset
- Split seed: `42`
- Augmentation: horizontal flip, vertical flip, random rotation, random crop + resize, mild photometric augmentation, blur/noise-style augmentation
- Normalization:
  - image values scaled to `[0, 1]`
  - encoder normalization with ImageNet mean/std
    - mean: `[0.485, 0.456, 0.406]`
    - std: `[0.229, 0.224, 0.225]`

## Training configuration

- Batch size: `24`
- Epochs: `160`
- Optimizer: `AdamW`
- Learning rate: `3e-4`
- Minimum LR: `1e-6`
- Weight decay: `1e-4`
- Encoder weight decay: `0.05`
- New layer weight decay: `0.01`
- Scheduler: warmup cosine
- Warmup epochs: `5`
- Loss: `DiceBCEBoundaryLoss`
  - Dice weight: `0.55`
  - BCE weight: `0.25`
  - Boundary weight: `0.20`
  - Focal Tversky weight: `0.0`
  - Label smoothing: `0.02`
- AMP: enabled when CUDA is available
- Encoder freezing/unfreezing: `unfreeze_schedule none`
- Training seed: `42`

## Checkpoint selection

- Best checkpoint metric: validation IoU
- Validation set used: the held-out validation split from the training datasets
- Validation threshold: `0.45`
- External datasets do not affect checkpoint selection

## Evaluation

- Metrics: Dice, IoU, S-measure, Weighted F-measure, E-measure, MAE
- Prediction threshold: `0.45`
- TTA: supported in evaluation paths
- External datasets:
  - Kvasir-SEG
  - CVC-ClinicDB
  - CVC-300
  - CVC-ColonDB
  - ETIS-LaribPolypDB

## Exact runner script

- `ps_one_seed_ablation/32_fafem_layerwise_warmup_cosine_seed42.ps1`

## Key source files

- `main_torch.py`
  - training entry point
  - dataset split
  - augmentation
  - loss selection
  - optimizer/scheduler setup
  - checkpoint selection
  - training loop
- `models/convnext_pretrain.py`
  - `ConvNeXtUNet`
  - encoder/decoder construction
  - FAFEM placement
- `one_seed_models.py`
  - `build_experiment_model`
  - experiment-specific module enabling/disabling
- `ablation_registry.py`
  - experiment metadata for `one_seed_32_fafem_layerwise_warmup_cosine`
- `evaluate.py`
  - final best-checkpoint evaluation
- `evaluation_core.py`
  - metric computation and external dataset loaders

## Baseline comparison

- Baseline FAFEM reference: `one_seed_03_baseline_plus_fafem`
- The current live run differs from the baseline by:
  - no staged unfreezing
  - layerwise ConvNeXt optimizer grouping
  - warmup cosine scheduler
  - 160 epochs instead of the baseline training protocol

