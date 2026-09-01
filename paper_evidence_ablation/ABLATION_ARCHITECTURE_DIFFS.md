# Ablation Architecture Differences

Only code/registry-supported changes are stated here. FAFEM and MSCB are reused prior modules; the novel mechanism in this sequence is the FAFEM-conditioned frequency-guided coupling and its bounded residual form.

## Clean adjacent comparisons

### Exp33 -> Exp37

- Added one reused `MSCBLite` block after the normal Stage-3 decoder/skip fusion.
- FAFEM remains bottleneck-only; all other architecture flags remain off.

### Exp37 -> Exp43

- Replaced the unguided Stage-3 `MSCBLite` with `FrequencyGuidedMSCBLite`.
- Added three branch weights conditioned on the bottleneck FAFEM frequency descriptor.

### Exp43 -> Exp45

- Replaced direct frequency guidance with `ResidualFrequencyGuidedMSCBLite` using bounded interpolation between uniform and predicted branch weights.
- Set deterministic guidance initialization (`guidance_init_std=0.0`) and initial bounded strength `0.05`.
- Because Exp44 is incomplete, this comparison changes both residualization and initialization strength; it does not isolate them separately.

## Placement comparisons

### Exp37 -> Exp41

- Added the same unguided MSCB-lite block at Stage 2 while retaining Stage 3.

### Exp41 -> Exp42

- Added the same unguided MSCB-lite block at Stage 1 while retaining Stages 2 and 3.

### Exp45 -> Exp48/Exp49 (configured, not measured)

- Replaced the single post-fusion Stage-3 residual frequency-guided block with residual frequency-guided blocks on Stage-1, Stage-2, and Stage-3 skip tensors before fusion.
- Exp48 has no run directory. Exp49 has incomplete training artifacts and no evaluation, so no measured placement claim is available.

## Historical baseline context

- Old baseline -> old Exp03: added bottleneck FAFEM only.
- This architectural isolation is real, but its protocol is not comparable to Exp45 and is therefore excluded from the recommended final table.
