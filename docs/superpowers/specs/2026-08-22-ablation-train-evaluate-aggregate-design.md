# Ablation Train, Evaluate, and Aggregate Workflow

## Objective

Every `ps_ablation/<experiment>.ps1` runner executes an idempotent, sequential
three-seed workflow for seeds `42`, `6543`, and `7777`: train or resume only
when required, evaluate the best validation checkpoint, then aggregate the
three valid evaluation results. Existing optimizer, scheduler, loss,
augmentation, split, input size, and checkpoint-selection behavior remain
unchanged.

## Canonical Artifacts

Each experiment uses this layout:

```text
results/ablation/<experiment>/
  seed_42/
    best_checkpoint.pth
    training_history.csv
    training_summary.json
    evaluation_summary.json
    KvasirSEG-ConvNeXt_log.txt
  seed_6543/
  seed_7777/
  summary.csv
```

`best_checkpoint.pth` always contains the best-validation model weights, never
the final-epoch weights. Completion is metadata on that same checkpoint and
does not alter its model tensors.

The checkpoint payload retains the existing model, raw-model, EMA, optimizer,
and scheduler states and adds:

```text
experiment_name
seed
architecture
best_epoch
best_validation_metric
final_epoch
training_time_seconds
training_complete
```

Every improving checkpoint is written atomically with
`training_complete=false`. After a normal final epoch or normal early stop,
the same best checkpoint is atomically rewritten with unchanged weight/state
tensors and `training_complete=true`, the final epoch, cumulative training
time, and completion reason.

## Per-Seed State Machine

The shared PowerShell runner independently validates each seed before acting:

1. Missing checkpoint: start training.
2. Unloadable, malformed, wrong-seed, wrong-experiment, or architecture-
   incompatible checkpoint: stop with a clear error and preserve the file.
3. Valid checkpoint with `training_complete=false`: resume model, raw weights,
   EMA, optimizer, scheduler, epoch, elapsed time, and early-stopping state.
4. Valid checkpoint with `training_complete=true`: skip training.
5. Missing, malformed, schema-incomplete, wrong-seed, wrong-experiment, or
   wrong-checkpoint evaluation JSON: rerun evaluation from the completed best
   checkpoint.
6. Valid evaluation JSON: skip evaluation.
7. After all three seed evaluations validate, regenerate `summary.csv`.

Training or evaluation failure stops the experiment immediately. Aggregation
never runs on fewer than three valid seeds. A valid checkpoint is never
deleted or overwritten because evaluation failed.

Status output uses explicit messages such as:

```text
[seed 42] Valid completed checkpoint found - skipping training.
[seed 42] Evaluation missing or invalid - running evaluator.
[seed 42] Evaluation complete.
```

## Legacy Migration

Legacy epoch-named checkpoints are loaded and ranked by their stored
validation metric. Corrupt files cause a clear error and are preserved; they
are not silently ignored when they are the only evidence for a seed.

Completion is inferred only from reliable evidence:

- a valid legacy final checkpoint written by the trainer after normal loop
  exit or early stopping, with a matching final-save log entry; or
- a valid best checkpoint plus the exact trainer completion log
  `Training stopped after epoch <configured maximum>; best.pth retained.`,
  where the logged epoch equals the configured maximum epoch; or
- an existing structured training summary explicitly recording normal
  completion and matching the seed/experiment.

An epoch-named best checkpoint alone does not prove completion. When completion
is confirmed, the highest valid validation checkpoint becomes
`best_checkpoint.pth`, its best weights are preserved, completion metadata is
added, and superseded valid `.pth` files are removed only after the new file
round-trips successfully. When completion is unconfirmed, the highest valid
checkpoint becomes an incomplete resumable `best_checkpoint.pth`; training
resumes from it.

## Training Artifacts

`main_torch.py` receives optional artifact paths and experiment metadata so
existing non-ablation CLI usage stays compatible. The PowerShell workflow
supplies these arguments.

`training_history.csv` is append-safe across resumes, has one row per completed
epoch, and de-duplicates by epoch. It records epoch, training loss, validation
Dice, validation IoU, learning rates, elapsed seconds, and whether the row
improved the best metric.

`training_summary.json` is atomically updated and contains at least:

```json
{
  "experiment_name": "01_baseline",
  "seed": 42,
  "best_epoch": 37,
  "best_validation_metric": 0.8123,
  "final_epoch": 81,
  "training_time_seconds": 12345.6,
  "best_checkpoint_path": ".../best_checkpoint.pth",
  "training_complete": true,
  "completion_reason": "early_stopping"
}
```

Elapsed training time is cumulative across resumed invocations. An interrupted
run retains its history and incomplete best checkpoint but is not marked
complete.

## Shared Evaluation Contract

`evaluate.py` reconstructs the exact experiment architecture from one shared
Python ablation registry, validates checkpoint experiment/seed/architecture
metadata, strict-loads the scored best-validation weights, counts parameters,
and evaluates these dataset keys:

- `Kvasir-SEG`: the existing fixed 10% validation split.
- `CVC-ClinicDB`: the existing fixed 10% validation split.
- `CVC-300`: full external dataset.
- `CVC-ColonDB`: full external dataset.
- `ETIS-LaribPolypDB`: full external dataset stored at
  `data/ETIS-LARIBPOLYPDB`.

Evaluation is deterministic, uses input size `352x352`, no TTA, and no test-set
threshold search. `mDice` and `mIoU` are means of per-image binary metrics at
threshold `0.45`. `F_beta_w`, `S_alpha`, `mE_phi`, `maxE_phi`, and `MAE` use
continuous sigmoid probability maps through the existing `py_sod_metrics`
definitions. All experiments call this same implementation.

`evaluation_summary.json` is written atomically with:

```json
{
  "experiment_name": "01_baseline",
  "seed": 42,
  "checkpoint": ".../best_checkpoint.pth",
  "checkpoint_fingerprint": "sha256:...",
  "trainable_parameters": 123,
  "total_parameters": 456,
  "macs": 123456789,
  "flops": 246913578,
  "gmacs": 0.123456789,
  "gflops": 0.246913578,
  "results": {
    "Kvasir-SEG": {
      "mDice": 0.0,
      "mIoU": 0.0,
      "F_beta_w": 0.0,
      "S_alpha": 0.0,
      "mE_phi": 0.0,
      "maxE_phi": 0.0,
      "MAE": 0.0
    }
  }
}
```

All five dataset entries and all seven finite numeric metrics are required for
validity. The experiment, seed, and checkpoint SHA-256 must match current
artifacts before evaluation can be skipped.

Complexity is measured once per completed seed on the reconstructed model with
a single `1x3x352x352` input. The evaluator records exact integer `macs` and
`flops` plus `gmacs` and `gflops`, using the fixed convention
`flops = 2 * macs`. Parameter counts and complexity values are part of the
evaluation-validity schema; missing or non-finite values force reevaluation.

## Aggregation Contract

`summarize_seeds.py` requires exactly seeds `42`, `6543`, and `7777`, validates
each evaluation schema, and writes `summary.csv` atomically. Each dataset and
metric remains a separate row with `mean`, `sample_std`, and a formatted
`mean +/- sample_std` value. Sample standard deviation uses `ddof=1`
with `statistics.stdev`.

When parameter counts and complexity values are identical across seeds,
`summary.csv` records `trainable_parameters`, `total_parameters`, `macs`,
`flops`, `gmacs`, and `gflops` once in model-level rows with no mean or
standard deviation. Any mismatch is an architecture/reconstruction error and
stops aggregation.

## Implementation Boundaries

- Add `ablation_registry.py`, `evaluate.py`, and `summarize_seeds.py`.
- Extend `main_torch.py`, `checkpoint_management.py`, and the optional ablation
  CLI arguments only where structured/resumable artifacts require it.
- Update `ps_ablation/Invoke-Ablation.ps1`; individual experiment wrappers stay
  thin and retain their existing names.
- Reuse/refactor dataset loading from `evaluate_all_datasets.py` and advanced
  metrics from `main_torch.py` into one shared evaluation module rather than
  duplicating formulas.
- Do not change model layers, data splits, augmentation, optimizer, scheduler,
  loss, validation criterion, or external-test isolation.
- Do not launch a second GPU trainer, modify manuscript files, commit, or push.

## Verification

Automated tests cover:

- experiment registry mappings for every PowerShell wrapper;
- checkpoint schema, atomic save, integrity failure, legacy migration,
  incomplete resume, and completion metadata without weight mutation;
- append/deduplicate history and cumulative elapsed time;
- per-image Dice/IoU plus `py_sod_metrics` output schema on deterministic masks;
- fixed-input parameter/MAC/FLOP measurement with `flops = 2 * macs`;
- evaluator rejection of final/incompatible/corrupt checkpoints;
- evaluation JSON validation and checkpoint-fingerprint invalidation;
- three-seed sample standard deviation and identical-parameter handling;
- PowerShell dry-run branches for train, resume, evaluate-only, complete-skip,
  invalid checkpoint, invalid evaluation, and aggregate.

Target-environment verification uses the existing project venv and includes a
real checkpoint reconstruction/load plus a bounded evaluation smoke test. No
new full training campaign is launched during implementation verification.
