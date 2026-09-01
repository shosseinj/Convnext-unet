# Source Manifest

| Source | Purpose | Key evidence | Copied snapshot |
|---|---|---|---|
| `ablation_registry.py` | Immutable Exp.45 architecture contract | `ExperimentConfig`, Exp.45 registration | Yes |
| `one_seed_models.py` | Model constructor | `build_experiment_model` | Yes |
| `models/convnext_pretrain.py` | Encoder, decoder, FG-MSCB, model forward | `ConvNeXtUNet`, `ResidualFrequencyGuidedMSCBLite` | Yes |
| `models/fafem.py` | FAFEM and descriptor | `FrequencyAwareFeatureEnhancement` | Yes |
| `main_torch.py` | Dataset split, loss, optimizer, EMA, scheduler, training selection | `Dataset`, `KvasirSEGDataset`, `DiceBCEBoundaryLoss`, `create_warmup_cosine_scheduler` | Yes |
| `ps_one_seed_ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine_seed42.ps1` | Exact Exp.45 launch configuration | seed/LR/scheduler/module flags | Yes |
| `evaluate.py` | Final five-dataset evaluation entry point | strict checkpoint validation, threshold | Yes |
| `evaluation_core.py` | Evaluation loading/metrics/complexity | `evaluate_loader`, `binary_metrics_per_image` | Yes |
| `configs/splits/development_seed_42.json` | Available split artifact | file inventory; not selected as final split source because Exp.45 code uses `random_state=42` directly | Yes |
| `one_seed_results/.../seed_42/best_checkpoint.pth` | Final source of truth | selected metadata/state | No: 601,763,900-byte binary excluded |
| `.../training_summary.json` | Completion/best epoch | best epoch 184, final epoch 199 | Yes |
| `.../evaluation_summary.json` | Final five-dataset results | metrics, threshold, no TTA, complexity | Yes |
| `source_snapshots/final_checkpoint_metadata.json` | Portable checkpoint metadata extraction | architecture, optimizer profile, scheduler, best/final epoch | Yes; extracted without copying weights |
| `source_snapshots/training_log_config_excerpt.txt` | Compact training-log evidence | initialization, optimizer, AMP, clipping, final run state | Yes; 17 selected lines |
| `FINAL_MODEL_ARCHITECTURE.md` | Existing verified architecture report | complete Exp.45 architecture narrative | Copied to package root |

Intentionally excluded: model weights/checkpoints, datasets, predictions, full 138KB training log, and unrelated experiment code/results.
