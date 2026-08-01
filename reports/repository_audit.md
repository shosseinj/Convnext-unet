# Repository Audit

Date: 2026-08-01  
Repository: `ConvNeXt_Unet` (`main`, commit `8e7e691`)  
Workflow identity: `Convnext-unet-main`

## Status

The worktree was clean at audit time. The repository contains the complete local
copies of Kvasir-SEG (1000 pairs), CVC-ClinicDB (612 pairs), CVC-300 (60 pairs),
CVC-ColonDB (380 pairs), and ETIS-LARIBPOLYPDB (196 pairs). The ImageNet
ConvNeXt-Tiny weight file exists at `convnext_tiny_22k_1k_384.pth`; 229 training
checkpoints and 46 text logs were found below `logs/`.

## Active architecture

`main_torch.py` constructs `models.convnext_pretrain.ConvNeXtUNet`. It uses a
ConvNeXt-Tiny encoder, lightweight U-Net decoder, ImageNet normalization inside
the model, a three-dilation MSC module plus pooled branch, and `SimpleFusion`
skip blocks. The active code is therefore the `Baseline + MSC` variant. `BSEI`,
`DetailBranch`, and `GatedDetailFusion` exist but are not active. Deep-supervision
loss utilities exist, while the active model returns one logit tensor.

Project decision: `BSEI` is the only official module name. `SimpleFusion` is the
normal skip/fusion implementation. The source manuscript still contains a
legacy name that must be corrected during the controlled Word-update stage.

## Data and evaluation path

Images are loaded with OpenCV, converted BGR to RGB, resized to 352 by 352,
scaled to `[0, 1]`, and transposed to CHW. Masks use nearest-neighbor resize and
threshold 127. Kvasir and ClinicDB are independently split 90/10 with
`random_state=42`, then training and validation arrays are concatenated.

The current split indices are not persisted. Global Python, NumPy, PyTorch,
CUDA, and DataLoader seeds are not controlled as one reproducibility unit.
External datasets are selectable from the main runner and need an explicit
test-only guard.

## Protocol gaps

| Item | Required | Current audit finding |
|---|---|---|
| Maximum epochs | 150 | 50000 default |
| Encoder freeze | epochs 1-10 | decoder warmup 24 |
| Encoder LR | 1e-5 | derived as decoder LR times 0.1 |
| Decoder LR | 1e-4 | 1e-4 |
| Weight decay | 1e-4 | 5e-4; refine 1e-3 |
| Early stopping | 30 | 40 |
| Seeds | 42, 3407, 2026 | split seed 42 only |
| Ablation TTA | disabled | enabled by default |
| Best checkpoint | equal mean of two validation Dice values | combined validation IoU |
| Test isolation | external sets only after selection | no hard guard |
| Architecture selection | explicit config | manual active modules/comments |

The combined validation score is sample-weighted and is not equivalent to
`(kvasir_val_dice + clinicdb_val_dice) / 2`.

## Risks

1. Selection and threshold leakage through the main evaluation CLI.
2. Non-reproducible permutations and DataLoader sampling.
3. Architecture/checkpoint loaders that silently skip incompatible tensors.
4. Existing logs and checkpoints do not by themselves prove protocol compliance.
5. Paper values must not be imported unless linked to verified raw result files.

## Gate decision

Stage: Repository Audit  
Status: PASS  
Files changed: audit and design artifacts only  
Evidence: repository code, local dataset inventory, weights, checkpoints, logs  
Next gate: config-driven architecture inventory and Gate 1
