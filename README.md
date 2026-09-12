# ConvNeXt U-Net for Polyp Segmentation

This repository is an experimental framework for colorectal polyp segmentation using a ConvNeXt encoder within a U-Net-style architecture. It is used to study feature refinement, skip-fusion strategies, frequency-aware modules, and controlled architectural ablations across multiple public polyp datasets.

## Research Focus

The project investigates how modern ConvNeXt representations can be combined with lightweight decoder modules for accurate segmentation while preserving a reproducible experimental protocol. Variants explored in the repository include frequency-aware bottleneck processing, multi-scale convolution, attention/gating mechanisms, and decoder-stage refinement.

## Experimental Design

A common training protocol is used across ablation variants so that architectural changes can be compared under the same data split, optimization settings, checkpoint-selection rule, and evaluation procedure. See `ABLATION_TRAINING_PROTOCOL.md` for the experiment policy used by the project.

The repository includes support for the following polyp-segmentation datasets:

- Kvasir-SEG
- CVC-ClinicDB
- CVC-300
- CVC-ColonDB
- ETIS-LaribPolypDB

## Evaluation

Evaluation utilities support both quantitative and qualitative analysis. Depending on the experiment, reported metrics include Dice, IoU, precision, recall, specificity, pixel accuracy, structure-based measures, enhanced-alignment measures, weighted F-measure, and MAE.

The repository also contains tools for deterministic qualitative comparisons between ablation variants and for frame-level video evaluation.

## Video Inference

The video pipeline follows the same preprocessing used by the segmentation model. Frames are converted to RGB, resized to 352 × 352 for inference, and prediction masks are mapped back to the decoded frame resolution.

Example:

```bash
python inference_video.py \
  --input-video outputs/kvasir_input.mp4 \
  --checkpoint path/to/best_model.pth \
  --output-video outputs/kvasir_prediction.mp4 \
  --device cuda
```

When a frame-to-source manifest and ground-truth masks are supplied, the pipeline can export per-frame metrics and aggregate summaries.

## Reproducibility

The repository contains experiment-specific scripts, checkpoints/configuration conventions, evaluation utilities, and ablation documentation. Model/checkpoint mismatches are treated explicitly rather than silently ignored, and deterministic comparison utilities are provided for qualitative analysis.

## Related Work

The later RFG-MSCB research branch and manuscript are available in [RFG-MSCB](https://github.com/shosseinj/RFG-MSCB).
