# Qualitative ablation export

Arrange each prediction root with one folder per dataset, containing masks whose
filename stems match the source images. The same ID must exist in all three roots.

```text
predictions/baseline/Kvasir-SEG/<image_id>.png
predictions/second_ablation/Kvasir-SEG/<image_id>.png
predictions/third_ablation/Kvasir-SEG/<image_id>.png
```

Run from the repository root:

```powershell
python tools/generate_qualitative_ablation_examples.py `
  --data-root data `
  --baseline-root predictions/baseline `
  --second-ablation-root predictions/second_ablation `
  --third-ablation-root predictions/third_ablation
```

The default output is `qualitative_ablation_outputs/`. Ground truth is discovered
from `data/<dataset>/masks`; use `--ground-truth-root` only for a separate root.
Selection is deterministic and based on ground-truth area quantiles, not model
performance. Existing output is protected unless `--overwrite` is specified.

For the paper's seed-42 Exp01/Exp33/Exp45 comparison, first export the 30 model
predictions with validated checkpoint identities and evaluation preprocessing:

```powershell
..\.venv\Scripts\python.exe -m tools.export_qualitative_predictions
```

A JSON config may supply `data_root`, `baseline_root`, `second_ablation_root`,
`third_ablation_root`, `ground_truth_root`, and `output_root`; pass it with
`--config path/to/config.json`. CLI path arguments override config values.
