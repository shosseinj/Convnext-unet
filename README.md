## Ablation Training Protocol

Use `ABLATION_TRAINING_PROTOCOL.md` for all ConvNeXtUNet ablation runs so every variant uses the same fair training setup and checkpoint-selection rule.

Edge enhancement ->

- Depthwise conv → Sigmoid, multiplied into skip
- Sharpen boundaries

SE (Squeeze-Excitation)

- Channel attention via global avg pool + MLP
- Reweight channels globally

Multi-scale pooled cross-attention

- Decoder queries skip at pool sizes (4, 8)
- Let decoder features guide which skip info to use

Learned gate

- Sigmoid over [skip, ctx] concat
- Control how much attention context to blend in

Dice = 0.9114, IoU = 0.8397 at threshold 0.30
The BSEI debug stats show the gate is ~0.5 (active), and the ctx/skip ratio is 0.55–0.83, meaning the attention context is meaningfully contributing.

CVC-300 -> 61
ETIS-LARIBPOLYPDB -> 196
CVC-COLONDB -> 380
CVC-ClinicDB -> 61 Test and 551 Train
Kvasir-SEG -> 100 Test and 900 Train

## Video Evaluation

The video tools reuse the validation preprocessing in this repository: OpenCV BGR
frames are converted to RGB, resized to `352 x 352`, scaled to `[0, 1]`, and
passed to the model. `ConvNeXtUNet` applies ImageNet normalization internally.
Training behavior is not affected.

Expected Kvasir-SEG layout:

```text
data/Kvasir-SEG/
  images/   # .jpg, .jpeg, or .png
  masks/    # masks with matching filename (or matching stem)
```

Create an input video and a JSON frame-to-source manifest:

```bash
python tools/create_kvasir_video.py --images-dir data/Kvasir-SEG/images \
  --ground-truth-dir data/Kvasir-SEG/masks --output outputs/kvasir_input.mp4 \
  --fps 10 --width 640 --height 480 --frames-per-image 3
```

Use `--letterbox` to preserve aspect ratio, `--shuffle --seed 42` for a
repeatable shuffled sequence, `--max-images N` for a subset, and `--repeat N`
to repeat the selected sequence. The default manifest is
`outputs/kvasir_input_manifest.json`.

Run segmentation while preserving the source resolution and FPS:

```bash
python inference_video.py --input-video outputs/kvasir_input.mp4 \
  --checkpoint path/to/best_model.pth --output-video outputs/kvasir_prediction.mp4 \
  --device cuda --threshold 0.5 --min-area 100
```

The default model is `models.convnext_pretrain:ConvNeXtUNet`. If construction
requires local encoder initialization, add
`--encoder-weights convnext_tiny_22k_1k_384.pth`. Direct state dictionaries and
checkpoint keys named `state_dict` or `model_state_dict` are supported, as are
DataParallel `module.` prefixes. Model/checkpoint mismatches fail explicitly.

Ground-truth contours, side-by-side output, and metrics:

```bash
python inference_video.py --input-video outputs/kvasir_input.mp4 \
  --checkpoint path/to/best_model.pth --output-video outputs/kvasir_prediction.mp4 \
  --manifest outputs/kvasir_input_manifest.json \
  --ground-truth-dir data/Kvasir-SEG/masks --show-ground-truth --side-by-side \
  --metrics-csv outputs/frame_metrics.csv --metrics-json outputs/summary_metrics.json
```

Outputs include the annotated MP4, per-frame CSV (Dice, IoU, precision, recall,
specificity, and pixel accuracy), and aggregate JSON. `--display` provides a
preview and `q` stops processing; omit it on headless systems. Other controls
include `--mask-alpha`, `--contour-thickness`, `--box-thickness`, `--output-fps`,
and `--codec`.

Troubleshooting:

- Checkpoint mismatch: select the same `--model` architecture used for training;
  the error lists missing, unexpected, or size-mismatched weights.
- CUDA unavailable/out of memory: use `--device cpu`, or verify the installed
  PyTorch build and GPU driver.
- MP4 writer failure: try `--codec mp4v` and ensure OpenCV has video codec support.
- Incorrect mask dimensions: keep the default `--input-size 352`; probability
  masks are always resized back to the decoded frame dimensions before drawing.

## Qualitative ablation examples

Generate two deterministic, common-ID comparisons per dataset from three
prediction roots:

```powershell
python tools/generate_qualitative_ablation_examples.py --data-root data `
  --baseline-root predictions/baseline --second-ablation-root predictions/second_ablation `
  --third-ablation-root predictions/third_ablation
```

See `docs/QUALITATIVE_ABLATION.md` for the input layout, JSON-config support,
ground-truth panels, validation behavior, and overwrite protection.

The helper `tools/export_qualitative_predictions.py` prepares the seed-42
Exp01/Exp33/Exp45 masks from their validated checkpoints.
