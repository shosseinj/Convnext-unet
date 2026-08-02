# Review scope

Independent review of the canonical seed-42 UGBR pilot campaign in `results/pilot_campaigns/ugbr_seed42_v2`. No training was rerun and the orchestration validator was not rerun.

# Pilot matrix

| Variant | Exact output directory | Seed | Epochs | Best selection Dice | Validator |
|---|---|---:|---:|---:|---|
| baseline | `results/pilot_campaigns/ugbr_seed42_v2/baseline/seed_42/pilot` | 42 | 12 | 0.078797 | PASS |
| baseline_ugbr | `results/pilot_campaigns/ugbr_seed42_v2/baseline_ugbr/seed_42/pilot` | 42 | 12 | 0.293464 | PASS |
| baseline_best_existing | `results/pilot_campaigns/ugbr_seed42_v2/baseline_best_existing/seed_42/pilot` | 42 | 12 | 0.282087 | PASS |
| baseline_best_existing_ugbr | `results/pilot_campaigns/ugbr_seed42_v2/baseline_best_existing_ugbr/seed_42/pilot` | 42 | 12 | 0.317964 | PASS |

# Evidence checked

For each run, `resolved_config.json`, `history.json`, `summary.json`, `best.pth`, and `last.pth` were read directly. All required files were present; histories contained epochs 1-12; summaries were complete and agreed with histories and checkpoints; checkpoint payloads loaded successfully; metadata included variant, seed, protocol, manifest, source fingerprints, environment, and pretrained-weight fingerprint; all inspected numeric metrics were finite. The queue log independently records four fresh runs in the canonical order and four validator PASS results. The four output directories are disjoint, all checkpoint hashes are distinct, and no pilot path is under `results/raw` or an official output namespace. No copied, aliased, resumed, or legacy-read artifact was found.

# Per-run verdict

- `baseline`: PASS. Identity, seed, namespace, 12-epoch history, metrics, checkpoints, metadata, summary, and validator evidence agree.
- `baseline_ugbr`: PASS. Same technical checks pass; paired UGBR result is internally consistent.
- `baseline_best_existing`: PASS. Same technical checks pass; the pilot namespace is separate from the existing source candidate.
- `baseline_best_existing_ugbr`: PASS. Same technical checks pass; paired UGBR result is internally consistent.

# Cross-run consistency

All runs use the same commit, development manifest, protocol, pretrained-weight hash, pilot limits, and seed. Variant-specific model parameter counts and checkpoint contents differ as expected. Summary Dice equals the best checkpoint and the maximum history Dice in every run. The paired Dice gains are `0.293464 - 0.078797 = 0.214668` and `0.317964 - 0.282087 = 0.035877`; no required paired safety metric materially degrades.

# Promotion criteria

`AGENTS.md` documents promotion only when paired UGBR gain is at least `0.003` absolute Dice and no required safety metric materially degrades. The workflow documents seed-42 pilot review before full experiments. The pilot protocol fixes seed 42 and 12 pilot epochs. No additional numeric criterion was invented.

# Gate decision

PASS. Both paired comparisons meet the documented gain gate, all four pilots are technically valid, and the evidence is ready for the next campaign stage. This establishes pilot readiness, not final scientific superiority or manuscript eligibility.

# Exact blockers, if any

None for this gate. `.agentic/state.json` was absent at review start and is created by this decision record; no training process is active.

# Recommended next stage

Promote to `EXPERIMENT`. Do not launch in this command. Launch the newly approved stage through `/resume-workflow`.
