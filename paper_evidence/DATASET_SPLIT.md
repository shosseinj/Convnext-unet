# Dataset Split and Preprocessing

## Observed data counts and use

Counts below were measured from the local `data/<dataset>/images` directories. Dataset loading accepts only valid image/mask filename pairs, so these are source-directory counts; the final evaluation summaries confirm 100/62/60/380/196 evaluated samples.

| Dataset | Total images | Train | Validation | External test | Use |
|---|---:|---:|---:|---:|---|
| Kvasir-SEG | 1000 | 900 | 100 | — | Training and held-out validation; external evaluation uses the same 100 held-out images |
| CVC-ClinicDB | 612 | 550 | 62 | — | Training and held-out validation; external evaluation uses the same 62 held-out images |
| CVC-300 | 60 | — | — | 60 | External evaluation only |
| CVC-ColonDB | 380 | — | — | 380 | External evaluation only |
| ETIS-LaribPolypDB | 196 | — | — | 196 | External evaluation only |

`main_torch.py:Dataset.get_features_vectors` independently applies `train_test_split(..., test_size=0.1, random_state=42, shuffle=True)` to Kvasir-SEG and CVC-ClinicDB, then concatenates their training subsets. The combined validation subset is shuffled and is the only checkpoint-selection set. External datasets do not influence training or checkpoint selection.

## Preprocessing and augmentation

- Image files: PNG/JPEG/TIF variants; BGR read by OpenCV then converted to RGB.
- Images: `cv2.INTER_LINEAR` resize to 352x352, converted to float in [0,1], then CHW.
- Masks: `cv2.INTER_NEAREST` resize and binarization `mask > 127`.
- Training augmentations in `KvasirSEGDataset`, each applied independently: horizontal flip (p=0.5); vertical flip (p=0.5); rotation uniformly sampled from -15 to +15 degrees (p=0.5; bilinear image / nearest-neighbor mask); crop with a uniformly sampled 0.7--1.0 crop ratio then resize back to 352x352 (p=0.5; bilinear image / nearest-neighbor mask); brightness and contrast factors 0.85--1.15 plus saturation 0.9--1.1 (all together, p=0.5); Gaussian blur with kernel 3 (p=0.15); and additive Gaussian noise with standard deviation 0.015 (p=0.25). Image values are clamped to [0,1] and masks are binarized at `>0.5` before return. The commented affine, elastic, and cutout blocks are not active.
- Model-side encoder normalization: ImageNet mean `(0.485,0.456,0.406)` and standard deviation `(0.229,0.224,0.225)`.

For external evaluation, `evaluation_core.read_dataset` repeats only deterministic RGB conversion, 352x352 resize, [0,1] scaling and binary mask conversion; it does not apply training augmentation.
