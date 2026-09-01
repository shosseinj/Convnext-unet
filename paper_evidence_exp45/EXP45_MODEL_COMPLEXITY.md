# Exp.45 Model Complexity

## Architecture

`one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine` is a ConvNeXt-Tiny U-Net with FAFEM at the bottleneck and one residual frequency-guided MSCB-lite at the Stage-3 decoder/skip-fusion position. All three completed seed checkpoints record the same architecture metadata.

## Counts and complexity

| Quantity | Value | Evidence |
|---|---:|---|
| Trainable parameters | 30,057,415 | Each final `evaluation_summary.json`; `evaluation_core.count_parameters` |
| Total parameters | 30,057,415 | Each final `evaluation_summary.json`; `evaluation_core.count_parameters` |
| MACs | 13,909,244,496 | Each final `evaluation_summary.json` |
| GMACs | 13.909244496 | Each final `evaluation_summary.json` |
| FLOPs | 27,818,488,992 | Each final `evaluation_summary.json` |
| GFLOPs | 27.818488992 | Each final `evaluation_summary.json` |
| Complexity input | 1 x 3 x 352 x 352 | `evaluation_core.measure_complexity` |

## Method and limitation

The project’s existing `evaluation_core.measure_complexity` function uses `thop.profile` on a deep-copied model in evaluation mode with a zero `1 x 3 x 352 x 352` input. It receives MACs from THOP, then defines FLOPs as `2 * MACs`. Thus, the reported 27.818488992 GFLOPs is explicitly FLOPs under this conversion convention, not MACs mislabeled as FLOPs. Complexity is architecture-level and identical for the three seeds; it excludes data loading and post-processing.
