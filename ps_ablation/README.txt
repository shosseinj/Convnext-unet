Polyp Segmentation Ablation Launchers

These launchers call the repository-root main_torch.py through
Invoke-Ablation.ps1. They use the shared 352x352 ablation protocol and the
canonical seeds 42, 3407, and 2026. Seeds run sequentially, never concurrently.

Run only the seed-42 baseline:
  .\ps_ablation\01_baseline.ps1 -Seeds 42

Run all canonical seeds for the baseline:
  .\ps_ablation\01_baseline.ps1

Inspect the generated command without training:
  .\ps_ablation\01_baseline.ps1 -Seeds 42 -DryRun

Results and checkpoints are written beneath:
  results\ablation\<experiment>\seed_<seed>\

Each seed keeps only checkpoints_KvasirSEG-ConvNeXt\best.pth. On restart, the
launcher automatically resumes model, EMA, optimizer, and scheduler state from
that file. Legacy epoch-named checkpoints are validated and consolidated to the
highest-scoring valid checkpoint before training resumes.

Experiment 11 has the same architecture as experiment 05 and normally should
reuse experiment 05 results rather than retraining.
