# Qualitative Results Provenance Report

## Final comparison

- Common image set: 6 cases (3 Kvasir-SEG, 3 CVC-ClinicDB).
- Column order: Input, GT, Baseline, CTNet, MEGANet, EnFormer, Ours (Exp45).
- Masks: binary white foreground on black background; nearest-neighbor mask resizing only.
- Exp45/Baseline inference: 352 x 352 input, threshold 0.45, no TTA, repository-native construction and preprocessing.
- External predictions: author-released precomputed maps; no external training or random weights.

| Model | Official repository | Checkpoint or prediction-map source | Exact checkpoint/artifact | Inference run? | Dataset | Selected predictions | Included? | Notes |
|---|---|---|---|---|---|---:|---|---|
| Baseline | This repository | Verified seed-42 checkpoint | `one_seed_results/ablation/01_baseline/seed_42/best_checkpoint.pth` | Yes | Kvasir-SEG, CVC-ClinicDB | 6 | Yes | Strict metadata, architecture, and state-dict validation; epoch 120. |
| CTNet | https://github.com/Fhujinwu/CTNet | Author repository `result_map.zip` | `external_models/CTNet/result_map.zip` | No | Kvasir, CVC-ClinicDB | 6 | Yes | Official archive contains 100 Kvasir and 62 ClinicDB maps. |
| MEGANet | https://github.com/UARK-AICV/MEGANet | Author-linked MEGANet-Res2Net precomputed maps | `external_models/_official_artifacts/MEGANet_Res2Net_predictions.zip` | No | Kvasir, CVC-ClinicDB | 6 | Yes | Exact Res2Net prediction-map variant requested. |
| EnFormer | https://github.com/HuangDLab/EnFormer | Author-linked precomputed maps | `external_models/_official_artifacts/EnFormer_prediction_maps.zip`, `result_map/res/enformer/` | No | Kvasir, CVC-ClinicDB | 6 | Yes | Included because the official archive was reproducibly downloadable and complete. |
| Ours (Exp45) | This repository | Verified seed-42 best checkpoint | `one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_42/best_checkpoint.pth` | Yes | Kvasir-SEG, CVC-ClinicDB | 6 | Yes | Experiment ID `one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine`; best epoch 184; validation-IoU selection value 0.878051; 30,057,415 parameters. |
| SEPNet | https://github.com/wangtong627/SEPNet | Author-linked OneDrive predictions/checkpoint | OneDrive prediction-map link in official README | No | - | 0 | No | Direct author link returned an invalid 13-byte response in this environment; no maps were used. |
| RAPUNet | https://github.com/hyunnamlee/RAPUNet | Author-linked Naver pretrained models/predicted images | Naver links in official README | No | - | 0 | No | Published prediction export is index-based; a trustworthy mapping from result indices to the selected source image IDs was not established. |
| EAT | https://github.com/deepang-ai/EAT | Official Hugging Face checkpoints | `deepang/eat` | No | - | 0 | No | Official inference requires its CUDA-compiled DCNv4 extension; it was not compiled into the Exp45 environment and no official precomputed maps were published. |

## Selected images

| Dataset | Image ID | Case type |
|---|---|---|
| Kvasir-SEG | `cju2rmd2rsw9g09888hh1efu0.jpg` | Small polyp |
| CVC-ClinicDB | `25.png` | Large polyp |
| Kvasir-SEG | `cju31w6goazci0799n014ly1q.jpg` | Low contrast |
| CVC-ClinicDB | `374.png` | Blurred or ambiguous boundary |
| Kvasir-SEG | `cju2yo1j1v0qz09934o0e683p.jpg` | Irregular polyp |
| CVC-ClinicDB | `80.png` | Confusing background / difficult case |

## Integrity and reproducibility

Full source paths, archive SHA-256 hashes, checkpoint SHA-256 hashes, inference settings, and row order are recorded in `qualitative_results_provenance.json`. The build is reproducible with `tools/build_final_qualitative_comparison.py`. External continuous-valued maps were binarized at 128 only for the requested binary visual presentation; their original author archives remain preserved. No claim is made that externally generated maps used the same training split or threshold as Exp45.

The main PNG is 4800 x 3720 at 400 dpi. The PDF is one page and was rendered back to PNG for visual inspection. Both visual checks showed aligned rows, readable labels, binary masks, and no `N/A` cells.
