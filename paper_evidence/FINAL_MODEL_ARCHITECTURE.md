# Final Model Architecture

## Scope and source of truth

This report describes the model selected by `one_seed_results/ablation/45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine/seed_7777/best_checkpoint.pth`. The checkpoint's `architecture` metadata agrees with `ablation_registry.py:get_experiment()` for `one_seed_45_fafem_residual_frequency_guided_mscb_stage3_stronger_init_warmup_cosine`. Model construction follows `one_seed_models.py:build_experiment_model()` into `models/convnext_pretrain.py:ConvNeXtUNet`.

Enabled architectural options are: ConvNeXt-Tiny backbone, bottleneck FAFEM, residual frequency-guided MSCB-lite at Stage 3, normal skip fusion, bilinear upsampling, and a 96-channel high-resolution decoder. MSC, BSEI, detail branch/GDF, CSAF, stage-specific FAFEMs, UGBR, MixStyle, deformable variants, ordinary MSCB-lite, and deep supervision are disabled.

## 1. Encoder

`models/convnext_pretrain.py:ConvNeXtEncoder` receives an ImageNet-normalized RGB tensor, using mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. Its ConvNeXt-Tiny depths are `[3, 3, 9, 3]`, dimensions are `[96, 192, 384, 768]`, drop-path rate is `0.25`, and feature dropout is `0.2` after encoder stages 2--4.

| Feature | Construction | General shape | Shape for 352x352 input |
|---|---|---:|---:|
| `f1` | 4x4, stride-4 stem convolution + LayerNorm + 3 ConvNeXt blocks | `96 x H/4 x W/4` | `96 x 88 x 88` |
| `f2` | LayerNorm + 2x2, stride-2 convolution + 3 blocks | `192 x H/8 x W/8` | `192 x 44 x 44` |
| `f3` | LayerNorm + 2x2, stride-2 convolution + 9 blocks | `384 x H/16 x W/16` | `384 x 22 x 22` |
| `f4` | LayerNorm + 2x2, stride-2 convolution + 3 blocks | `768 x H/32 x W/32` | `768 x 11 x 11` |

Each `models/convnext_pretrain.py:Block` is a 7x7 depthwise convolution followed by channels-last LayerNorm, a `C -> 4C -> C` pointwise MLP with GELU, learnable layer scale initialized to `1e-6`, stochastic depth, and a residual addition.

## 2. Bottleneck data flow

The exact path is:

`f4 -> FAFEM -> enhanced f4 + 1536-D frequency descriptor -> LiteBottleneck -> decoder4`.

`models/convnext_pretrain.py:LiteBottleneck` preserves `768 x H/32 x W/32`: `1x1 Conv 768->192 -> LayerNorm -> GELU -> 3x3 depthwise-separable Conv 192->192 -> LayerNorm -> GELU -> Dropout2d(0.2) -> 1x1 Conv 192->768 -> LayerNorm`, followed by `f4 + gamma_b * transform(f4)`. `gamma_b` is a learned scalar initialized to zero. `enable_msc=false`, so `context` is `nn.Identity`; `MultiScaleContext` is not in this final model.

## 3. FAFEM placement and operation

Only bottleneck FAFEM is enabled. `models/fafem.py:FrequencyAwareFeatureEnhancement` is applied to raw `f4` before `LiteBottleneck`; `fafem_stage1`, `fafem_stage2`, and `fafem_stage3` are absent.

For `x=f4`, FAFEM computes `low=AvgPool3x3(x)` and `high=x-low`. Separate 3x3 depthwise-convolution + GELU branches enhance low and high components. Global pooling of their concatenation produces the `2C=1536` frequency descriptor used later by Stage-3 guidance. A channel-wise gating subnetwork (`AdaptiveAvgPool -> 1x1 Conv 1536->48 -> GELU -> 1x1 Conv 48->1536 -> sigmoid`) supplies separate low/high weights. Their weighted sum passes through a 3x3 depthwise refinement, and FAFEM returns `x + gamma_f * refined`, where `gamma_f` is learned and initialized to zero.

## 4. Stage-3 fusion

Here “Stage 3” means fusion with encoder feature `f3` at `H/16`, producing decoder feature `d4`:

1. `decoder4` bilinearly upsamples the 768-channel bottleneck feature by 2, then applies depthwise-separable 3x3 convolution `768->384`, LayerNorm, GELU, and Dropout2d(0.2).
2. The result is resized to `f3` if necessary and concatenated with the unchanged 384-channel `f3`, yielding 768 channels.
3. Because `skip_mode="normal"`, `bsei4` is actually `models/convnext_pretrain.py:SimpleFusion`, not BSEI: `1x1 Conv 768->384 -> LayerNorm -> GELU -> depthwise-separable 3x3 Conv 384->384 -> LayerNorm -> GELU -> Dropout2d(0.2)`.
4. `ResidualFrequencyGuidedMSCBLite` then refines this fused 384-channel `d4` using the FAFEM descriptor. Output remains `384 x H/16 x W/16` (`384 x 22 x 22` at 352x352).

There is no Stage-3 attention gate, CSAF, cross-level fusion, geometry convolution, LKA, or direct FAFEM on `f3`.

## 5. MSCB and residual frequency-guided MSCB

`models/convnext_pretrain.py:MSCBLite` implements the reused EMCAD-style MSCB topology: `1x1 Conv C->2C -> BatchNorm -> ReLU6`; three parallel depthwise branches with kernels 1, 3, and 5, each followed by BatchNorm and ReLU6; branch summation; channel shuffle; `1x1 Conv 2C->C -> BatchNorm`; and residual addition to the block input.

The final model does **not** instantiate ordinary `MSCBLite`. It instantiates `models/convnext_pretrain.py:ResidualFrequencyGuidedMSCBLite` once, after Stage-3 normal skip fusion. For `C=384`, its expanded width is 768. Its guidance MLP maps the 1536-D FAFEM descriptor through `1536->48->3`; `3 * softmax(logits)` produces per-sample weights `alpha_1, alpha_3, alpha_5` whose sum is 3. If `B_k` denotes a kernel branch:

`ordinary = sum_k B_k`, `guided = sum_k alpha_k B_k`, and `mixed = ordinary + s * (guided - ordinary)`.

The effective strength is the learned scalar `s` clamped to `[0, 0.5]` (unsigned mode). This selected configuration initializes `s=0.05`; the final guidance-layer weights use `guidance_init_std=0.0` and its bias is zero, so initial softmax weights are uniform and the initial guided and ordinary sums coincide. `mixed` then undergoes channel shuffle, `1x1 Conv 768->384`, BatchNorm, and residual addition to the Stage-3 input.

## 6. Remaining decoder and skips

All later skips use `SimpleFusion`; no BSEI or attention gates are active.

| Output | Data flow | Shape for 352x352 |
|---|---|---:|
| `d3` | `decoder3(d4)`: bilinear x2 + separable 3x3 `384->192`; concatenate `f2` (192); `SimpleFusion 384->192` | `192 x 44 x 44` |
| `d2` | `decoder2(d3)`: bilinear x2 + separable 3x3 `192->96`; concatenate `f1` (96); `SimpleFusion 192->96` | `96 x 88 x 88` |
| `d1` | `decoder1(d2)`: bilinear x2 + separable 3x3 `96->96`; no encoder skip; `SimpleFusion 96->96` | `96 x 176 x 176` |

No MSCB is applied at `d3` or `d2`, and there is no detail branch/fusion at `d1`.

## 7. Deep supervision

Disabled: `deep_supervision_heads=0`, so `ConvNeXtUNet.auxiliary_heads` is empty and `forward()` returns only the main logits. No auxiliary 1x1 heads are instantiated.

## 8. Final prediction head

`models/convnext_pretrain.py:ConvNeXtUNet.final_refine` applies: depthwise-separable 3x3 `96->96 -> LayerNorm -> GELU -> depthwise-separable 3x3 96->48 -> LayerNorm -> GELU -> 1x1 Conv 48->1`. The single-channel logits at `H/2 x W/2` are bilinearly resized to the original input size with `align_corners=False`. Sigmoid is not applied inside the model.

## 9. Trainable parameters

**30,057,415 trainable parameters** (and 30,057,415 total parameters), as recorded by `evaluation_core.py:count_parameters()` in this checkpoint's `evaluation_summary.json`. The evaluation path reconstructs the registry architecture, loads `model_state_dict` with `strict=True`, and then counts parameters.

## 10. Reused versus novel components

### Existing/reused building blocks

- ConvNeXt-Tiny encoder and ConvNeXt block topology.
- U-Net-style progressive decoder and encoder skip concatenations.
- FAFEM (`FrequencyAwareFeatureEnhancement`) as an existing frequency-aware enhancement module.
- EMCAD-style MSCB topology (`MSCBLite`): pointwise expansion, parallel multi-kernel depthwise branches, channel shuffle, projection, and residual connection.
- Standard bilinear upsampling, LayerNorm/GELU decoder blocks, and the existing `LiteBottleneck`/`SimpleFusion` infrastructure.

### Mechanism introduced in this work

- `ResidualFrequencyGuidedMSCBLite`: the bottleneck FAFEM low/high descriptor is reused across network levels to generate per-image weights for the three MSCB scale branches.
- The bounded residual interpolation `ordinary + s(guided-ordinary)`, rather than replacing ordinary MSCB outright. It preserves the ordinary MSCB path at uniform guidance/zero strength while allowing learned frequency-conditioned deviation.
- Its placement specifically after the first decoder/`f3` fusion, coupling global bottleneck frequency evidence to multi-scale refinement at `H/16`.

The paper should therefore avoid claiming FAFEM or MSCB themselves as new. The defensible architectural novelty is their cross-level coupling through the FAFEM descriptor, the residual/bounded branch-guidance formulation, and its Stage-3 placement.

## Relevant implementation locations

- `ablation_registry.py`: `ExperimentConfig`, `get_experiment`, selected experiment tuple.
- `one_seed_models.py`: `build_experiment_model`.
- `models/convnext_pretrain.py`: `ConvNeXtEncoder`, `Block`, `LiteBottleneck`, `DecoderBlock`, `SimpleFusion`, `MSCBLite`, `FrequencyGuidedMSCBLite`, `ResidualFrequencyGuidedMSCBLite`, `ConvNeXtUNet.forward`, `ConvNeXtUNet.final_refine`.
- `models/fafem.py`: `FrequencyAwareFeatureEnhancement.forward_with_frequency_descriptor`.
- `evaluate.py`: `build_model` and strict checkpoint loading.
- `evaluation_core.py`: `count_parameters`.
