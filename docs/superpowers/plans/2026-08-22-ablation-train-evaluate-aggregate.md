# Ablation Train, Evaluate, and Aggregate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not dispatch subagents unless the user explicitly requests delegation.

**Goal:** Make every ablation PowerShell runner idempotently train/resume, evaluate the best validation checkpoint on five datasets, and aggregate seeds `42`, `6543`, and `7777` with mean plus sample standard deviation.

**Architecture:** A shared Python registry owns experiment architecture definitions. Structured checkpoint/artifact helpers provide atomic, schema-validated state; `main_torch.py` emits resumable training artifacts without changing optimization behavior; one evaluator owns all metric definitions and complexity measurement; one summarizer aggregates validated seed JSON; the shared PowerShell runner orchestrates the artifact state machine.

**Tech Stack:** Python 3.10, PyTorch, NumPy, OpenCV, `py_sod_metrics`, THOP, `unittest`, PowerShell.

**Spec:** `docs/superpowers/specs/2026-08-22-ablation-train-evaluate-aggregate-design.md`

## Global Constraints

- Use exactly seeds `42`, `6543`, and `7777`, sequentially.
- Keep `352x352`, current splits, augmentation, AdamW optimizer, scheduler, loss, validation-IoU selection criterion, and no-TTA evaluation unchanged.
- Use threshold `0.45` for per-image `mDice`/`mIoU`; use continuous sigmoid maps for the five SOD metrics.
- External datasets never affect checkpoint selection or thresholding.
- Preserve best-validation weights; completion is metadata only.
- Stop on corrupt/incompatible checkpoints; never silently overwrite them.
- Do not launch a full training campaign, modify manuscript files, commit, or push.
- Preserve unrelated dirty files and the existing `seed_3407` artifacts.

---

### Task 1: Centralize experiment definitions and artifact schemas

**Files:**
- Create: `ablation_registry.py`
- Create: `ablation_artifacts.py`
- Create: `tests/test_ablation_registry.py`
- Create: `tests/test_ablation_artifacts.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `get_experiment(name)`, `canonical_seeds()`.
- Produces: `checkpoint_sha256(path: Path) -> str`, `validate_checkpoint_metadata(checkpoint: dict, experiment: ExperimentConfig, seed: int) -> ValidationResult`, `validate_evaluation_summary(path: Path, experiment: str, seed: int, checkpoint_sha256: str) -> ValidationResult`, `atomic_write_json(path: Path, payload: dict) -> None`.

- [ ] **Step 1: Write failing registry tests**

```python
def test_canonical_seeds_are_requested_three_seed_set():
    assert canonical_seeds() == (42, 6543, 7777)

def test_full_model_registry_configuration():
    cfg = get_experiment("06_full_model")
    assert (cfg.enable_msc, cfg.skip_mode, cfg.detail_channels) == (True, "bsei", 32)
    assert (cfg.enable_gdf, cfg.detail_fusion_mode, cfg.deep_supervision_heads) == (True, "gdf", 3)
```

- [ ] **Step 2: Run the registry tests and confirm missing-module failure**

Run: `python -m unittest discover -s tests -p 'test_ablation_registry.py' -v`

Expected: import failure for `ablation_registry`.

- [ ] **Step 3: Implement the immutable registry**

```python
@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    enable_msc: bool
    skip_mode: str
    detail_channels: int
    enable_gdf: bool
    detail_fusion_mode: str
    deep_supervision_heads: int

SEEDS = (42, 6543, 7777)
EXPERIMENTS = {
    "01_baseline": ExperimentConfig("01_baseline", False, "normal", 0, False, "none", 0),
    "02_add_msc": ExperimentConfig("02_add_msc", True, "normal", 0, False, "none", 0),
    "03_add_lrse": ExperimentConfig("03_add_lrse", True, "bsei", 0, False, "none", 0),
    "04_add_db": ExperimentConfig("04_add_db", True, "bsei", 32, False, "concatenation", 0),
    "05_add_gdf": ExperimentConfig("05_add_gdf", True, "bsei", 32, True, "gdf", 0),
    "06_full_model": ExperimentConfig("06_full_model", True, "bsei", 32, True, "gdf", 3),
    "07_full_without_msc": ExperimentConfig("07_full_without_msc", False, "bsei", 32, True, "gdf", 3),
    "08_full_without_lrse": ExperimentConfig("08_full_without_lrse", True, "normal", 32, True, "gdf", 3),
    "09_gdf_concat": ExperimentConfig("09_gdf_concat", True, "bsei", 32, False, "concatenation", 3),
    "10_gdf_addition": ExperimentConfig("10_gdf_addition", True, "bsei", 32, False, "addition", 3),
    "11_full_without_ds": ExperimentConfig("11_full_without_ds", True, "bsei", 32, True, "gdf", 0),
}
```

Define all 11 existing wrapper names exactly once. Reject unknown names with a message listing valid names.

- [ ] **Step 4: Write failing artifact validation tests**

Cover valid JSON, malformed JSON, a missing dataset, a missing metric, non-finite metrics, wrong experiment/seed, and checkpoint SHA mismatch. Require these dataset names and metric names literally:

```python
DATASETS = ("Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")
METRICS = ("mDice", "mIoU", "F_beta_w", "S_alpha", "mE_phi", "maxE_phi", "MAE")
```

- [ ] **Step 5: Implement atomic JSON and validation helpers**

Define and return this type so PowerShell-facing commands can print precise status:

```python
@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str
```

Use sibling temporary files, parse the temporary JSON back, then `Path.replace()`.

- [ ] **Step 6: Run Task 1 tests**

Run: `python -m unittest discover -s tests -p 'test_ablation_registry.py' -v`

Run: `python -m unittest discover -s tests -p 'test_ablation_artifacts.py' -v`

Expected: all tests pass.

---

### Task 2: Implement completion-aware checkpoint and training artifacts

**Files:**
- Modify: `checkpoint_management.py`
- Modify: `ablation_cli.py`
- Modify: `main_torch.py`
- Create: `training_artifacts.py`
- Modify: `tests/test_checkpoint_management.py`
- Create: `tests/test_training_artifacts.py`

**Interfaces:**
- Consumes: `ExperimentConfig` and artifact atomic-write helpers from Task 1.
- Produces: `prepare_checkpoint(seed_dir: Path, experiment: ExperimentConfig, seed: int, max_epochs: int, log_path: Path) -> CheckpointDecision`, `mark_training_complete(checkpoint: dict, final_epoch: int, elapsed_seconds: float, reason: str) -> dict`, `append_history_row(path: Path, row: dict) -> None`, `write_training_summary(path: Path, payload: dict) -> None`.

- [ ] **Step 1: Write failing checkpoint-state tests**

Test missing, corrupt, incompatible, incomplete, complete, and legacy states. Prove completion metadata does not mutate tensors:

```python
before = checkpoint["model_state_dict"]["weight"].clone()
completed = mark_training_complete(checkpoint, final_epoch=149, elapsed_seconds=500.0, reason="max_epochs")
torch.testing.assert_close(completed["model_state_dict"]["weight"], before)
assert completed["training_complete"] is True
```

Legacy completion tests must accept only the spec’s three reliable evidence paths, including seed 42’s exact max-epoch log. A valid epoch-best checkpoint without completion evidence must produce `resume`, not `skip`.

- [ ] **Step 2: Run checkpoint tests and verify expected failures**

Run: `python -m unittest discover -s tests -p 'test_checkpoint_management.py' -v`

Expected: new completion and legacy tests fail.

- [ ] **Step 3: Generalize checkpoint management**

Define the state result explicitly:

```python
@dataclass(frozen=True)
class CheckpointDecision:
    action: Literal["train", "resume", "skip", "error"]
    checkpoint_path: Path | None
    reason: str
```

Canonicalize to `<seed_dir>/best_checkpoint.pth`. Keep `best.pth` and epoch-named files as legacy inputs only. Validate every candidate with `torch.load`; reject corrupt candidates with their path. Rank valid best candidates by stored validation metric. Atomically round-trip the canonical file before removing superseded valid legacy `.pth` files.

Checkpoint metadata must include experiment, seed, `asdict(ExperimentConfig)`, best/final epochs, cumulative time, early-stopping counter, completion reason, and completion flag. Strip only known THOP profiling keys ending in `.total_ops` or `.total_params` from legacy state dictionaries; reject all other architecture mismatches.

- [ ] **Step 4: Write failing history/summary tests**

```python
row = {
    "epoch": 4,
    "train_loss": 0.2,
    "validation_dice": 0.8,
    "validation_iou": 0.7,
    "encoder_lr": 1e-5,
    "decoder_lr": 1e-4,
    "elapsed_seconds": 12.0,
    "is_best": True,
}
append_history_row(path, row)
append_history_row(path, row)
rows = list(csv.DictReader(path.open()))
assert len(rows) == 1
assert rows[0]["epoch"] == "4"
assert rows[0]["is_best"] == "True"
```

Test atomic JSON, cumulative elapsed time across resumes, `best_epoch`, `final_epoch`, `best_validation_metric`, checkpoint path, and completion reason.

- [ ] **Step 5: Implement training artifact helpers and optional CLI**

Add optional CLI arguments: `--experiment_name`, `--seed_dir`, `--best_checkpoint_path`, `--training_history_path`, and `--training_summary_path`. Defaults preserve non-ablation usage.

Record one row immediately after each completed validation epoch. On improvement, atomically save best weights with `training_complete=false`. On normal max epoch or early stopping, rewrite only metadata in the best checkpoint and write the completed summary. On exceptions or process termination, never mark completion.

- [ ] **Step 6: Wire helpers into `main_torch.py` without changing training math**

Keep the existing forward/backward, optimizer, scheduler, loss, warmup, validation, and best-IoU comparison statements intact. Replace only artifact persistence and resume plumbing. Move complexity profiling out of the training model path so THOP buffers cannot enter future checkpoints.

- [ ] **Step 7: Run Task 2 and existing focused tests**

Run: `python -m unittest discover -s tests -p 'test_checkpoint_management.py' -v`

Run: `python -m unittest discover -s tests -p 'test_training_artifacts.py' -v`

Run: `python -m unittest discover -s tests -p 'test_ps_ablation_launchers.py' -v`

Expected: all pass.

---

### Task 3: Build one shared five-dataset evaluator

**Files:**
- Create: `evaluation_core.py`
- Create: `evaluate.py`
- Modify: `evaluate_all_datasets.py`
- Create: `tests/test_evaluation_core.py`
- Create: `tests/test_evaluate_cli.py`

**Interfaces:**
- Consumes: registry and artifact validation from Task 1; canonical completed checkpoint from Task 2.
- Produces: `evaluate_dataset(model, loader, device, threshold=0.45)`, `measure_complexity(model)`, and `evaluation_summary.json`.

- [ ] **Step 1: Write failing metric tests with hand-computed masks**

Use two images with different foreground sizes to prove metrics are averaged per image, not globally flattened. Assert `mDice`/`mIoU` literals. Feed continuous uint8 probability/GT pairs through `py_sod_metrics` and assert finite keys for all five SOD outputs.

- [ ] **Step 2: Run metric tests and confirm missing implementation**

Run: `python -m unittest discover -s tests -p 'test_evaluation_core.py' -v`

- [ ] **Step 3: Extract shared dataset and metric code**

Reuse `read_split`, the fixed random-state-42 validation split, loader construction, sigmoid output handling, and per-image Dice/IoU from `evaluate_all_datasets.py`. Move SOD calculator use from `main_torch.py` into `evaluation_core.py`; do not duplicate formulas. Preserve external dataset directory spelling while emitting the canonical JSON key.

- [ ] **Step 4: Write failing model reconstruction/checkpoint tests**

Construct tiny registry-compatible models where practical. Assert exact experiment configuration, scored `model_state_dict` loading, rejection of incomplete/final/wrong-seed checkpoints, legacy THOP-key sanitization, and parameter counts.

- [ ] **Step 5: Implement `evaluate.py`**

CLI:

```text
--experiment_name --seed --seed_dir --data_path --device --batch_size --num_workers
```

It must locate only `<seed_dir>/best_checkpoint.pth`, require `training_complete=true`, strict-load the scored best-validation weights after approved legacy sanitization, evaluate all five datasets with no TTA, and atomically write `<seed_dir>/evaluation_summary.json` with checkpoint SHA-256.

- [ ] **Step 6: Implement isolated complexity measurement**

Deep-copy the reconstructed model before THOP profiling so profiling buffers/hooks never touch the evaluation model or checkpoint. For input `(1, 3, 352, 352)`, record integer MACs/FLOPs and floating GMACs/GFLOPs with `flops = 2 * macs`; count trainable and total parameters from the unprofiled model.

- [ ] **Step 7: Make the legacy evaluator consume shared code**

Retain `evaluate_all_datasets.py` CLI compatibility but delegate dataset reading and metric primitives to `evaluation_core.py`, preventing future definition drift.

- [ ] **Step 8: Run Task 3 tests**

Run: `python -m unittest discover -s tests -p 'test_evaluation_core.py' -v`

Run: `python -m unittest discover -s tests -p 'test_evaluate_cli.py' -v`

Expected: all pass.

---

### Task 4: Aggregate exactly three validated seeds

**Files:**
- Create: `summarize_seeds.py`
- Create: `tests/test_summarize_seeds.py`

**Interfaces:**
- Consumes: evaluation-summary validator from Task 1.
- Produces: `<experiment_dir>/summary.csv`.

- [ ] **Step 1: Write failing aggregation tests**

Create three literal fixture summaries. Assert mean and `statistics.stdev` for a metric whose hand-calculated values are known. Assert 35 dataset/metric rows, six one-time complexity rows, rejection of missing seeds, and rejection of mismatched parameter/MAC/FLOP values.

- [ ] **Step 2: Run tests and confirm missing script failure**

Run: `python -m unittest discover -s tests -p 'test_summarize_seeds.py' -v`

- [ ] **Step 3: Implement deterministic CSV aggregation**

CLI: `--experiment_dir --experiment_name`. Read only `seed_42`, `seed_6543`, and `seed_7777`. Validate every JSON and checkpoint fingerprint. Write rows in registry dataset order and metric order with columns:

```text
row_type,dataset,metric,value,mean,sample_std,mean_plus_minus_std
```

Use `statistics.fmean` and `statistics.stdev`. Write complexity rows once with `value` populated and aggregation columns empty.

- [ ] **Step 4: Run Task 4 tests**

Run: `python -m unittest discover -s tests -p 'test_summarize_seeds.py' -v`

Expected: all pass.

---

### Task 5: Implement the idempotent PowerShell state machine

**Files:**
- Modify: `ps_ablation/Invoke-Ablation.ps1`
- Modify: `ps_ablation/01_baseline.ps1` through `11_full_without_ds.ps1`
- Modify: `ps_ablation/README.txt`
- Modify: `tests/test_ps_ablation_launchers.py`

**Interfaces:**
- Consumes: `main_torch.py`, `evaluate.py`, `summarize_seeds.py`, registry/state validation CLIs.
- Produces: sequential train/resume/evaluate/skip/aggregate behavior with clear status messages.

- [ ] **Step 1: Expand failing dry-run/state tests**

Use temporary experiment directories and a command-capture mode. Cover: no checkpoint→train; incomplete valid checkpoint→resume; completed checkpoint+missing evaluation→evaluate only; valid complete pair→skip; corrupt checkpoint→nonzero stop; incomplete JSON→evaluate; all seeds valid→summarize.

- [ ] **Step 2: Run PowerShell tests and verify failures**

Run: `python -m unittest discover -s tests -p 'test_ps_ablation_launchers.py' -v`

- [ ] **Step 3: Update shared orchestration**

Set default seeds to `@(42, 6543, 7777)`. Before every seed, call a lightweight Python state command that loads/validates the checkpoint and validates JSON schema/fingerprint. Branch exactly on returned state. Pass registry-consistent architecture flags and canonical artifact paths to training and evaluation. Check `$LASTEXITCODE` and required output existence after each subprocess.

Print the approved status format. Never treat a directory as completion. Never delete a checkpoint during evaluation handling. Run the summarizer only after revalidating all three seed evaluations.

- [ ] **Step 4: Keep wrappers thin and registry-checked**

Wrappers retain their filenames and experiment argument only. Tests compare every wrapper’s flags against `ablation_registry.py` so PowerShell/Python definitions cannot drift silently.

- [ ] **Step 5: Update operator documentation**

Document fresh, interrupted-training, evaluation-only, complete-skip, and corrupt-artifact behavior; list exact artifacts and seeds; include `-DryRun` examples. Explicitly state that `seed_3407` is legacy/out-of-scope and is preserved.

- [ ] **Step 6: Run all launcher dry runs**

Run each `ps_ablation/NN_*.ps1 -DryRun` against temporary state fixtures. Expected: no Python trainer/evaluator process starts; branch/status and command arrays match fixtures.

---

### Task 6: Migrate seed 42 and verify the integrated workflow without a campaign

**Files:**
- Modify artifacts only under: `results/ablation/01_baseline/seed_42/`
- Preserve: `results/ablation/01_baseline/seed_3407/`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: canonical completed seed-42 training artifacts and a bounded evaluation smoke/full evaluation only when the target venv supports it.

- [ ] **Step 1: Confirm no active trainer/evaluator**

Use `Get-CimInstance Win32_Process` and stop planning if any matching process exists. Do not terminate a process without user authorization.

- [ ] **Step 2: Back up migration inputs non-destructively**

Inventory filenames, sizes, hashes, checkpoint metadata, and the completion log. Do not copy multi-hundred-MB checkpoints unnecessarily; retain original files until canonical checkpoint round-trip and metadata checks succeed.

- [ ] **Step 3: Dry-run legacy inference**

Expected seed-42 evidence: valid nested `best.pth`, log line `Training stopped after epoch 150; best.pth retained.`, configured max epoch 150. Expected decision: completed legacy run, preserve best weights, create root `best_checkpoint.pth`, write history/summary from reliable available evidence, then remove superseded nested `.pth` only after validation.

- [ ] **Step 4: Execute and verify seed-42 migration**

Load the canonical checkpoint with the project venv, verify experiment/seed/architecture/completion metadata, compare pre/post best-weight tensor hashes, and confirm no unrelated artifacts changed.

- [ ] **Step 5: Run evaluator capability checks**

Run `evaluate.py --help`, reconstruct baseline, strict-load seed 42, and run a bounded one-batch-per-dataset smoke mode available only for verification. Confirm schema, metrics, parameter counts, and `flops = 2 * macs`. The smoke output must use a temporary path, not the canonical evaluation summary.

- [ ] **Step 6: Run the real seed-42 evaluation if smoke passes**

Evaluate all five datasets from `best_checkpoint.pth` and atomically write the canonical `evaluation_summary.json`. Validate its SHA and schema. Do not begin seed 6543 or 7777 training during implementation verification.

- [ ] **Step 7: Run fresh final verification**

Run focused tests for Tasks 1–5, `python -m py_compile` on all changed Python files, all 11 PowerShell dry runs, `git diff --check`, and inspect `git status --short`. Report full-suite failures separately if still caused by pre-existing missing `tools` or environment packages.

- [ ] **Step 8: Report exact changes and evidence**

List every added/modified source file, each migrated/generated seed-42 artifact, test counts, evaluation dataset counts/metrics, complexity convention, and any verification boundary. Leave all changes uncommitted.
