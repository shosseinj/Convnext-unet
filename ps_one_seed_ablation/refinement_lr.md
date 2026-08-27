# LR refinement runners

Run the five important seed-6543 ablations sequentially with cosine warm
restarts:

```powershell
.\ps_one_seed_ablation\Run-Important-Ablations-Cosine-Refinement-Seed6543.ps1 -BatchSize 24
```

The campaign loads each experiment's own `best_checkpoint.pth`, runs 56 extra
epochs with cycles of 8, 16, and 32 epochs, and writes to an isolated
`*_cosine_refinement/seed_6543` directory.

Run a single registered ablation when needed:

```powershell
.\ps_one_seed_ablation\Run-Ablation-LR-Refinement.ps1 `
  -Experiment one_seed_19_fafem_clfv2_fixed_unfreeze `
  -Seed 6543 `
  -BatchSize 24 `
  -AdditionalEpochs 56 `
  -LrScheduler cosine_warm_restarts `
  -CosineT0 8 `
  -CosineTMult 2 `
  -EarlyStopPatience 0 `
  -RefinementOutputName 19_fafem_clfv2_fixed_unfreeze_cosine_refinement
```

Use `-DryRun` to inspect commands without starting training.
