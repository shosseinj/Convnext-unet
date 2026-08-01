## Ablation Training Protocol

Use `ABLATION_TRAINING_PROTOCOL.md` for all ConvNeXtUNet ablation runs so every variant uses the same fair training setup and checkpoint-selection rule.

Edge enhancement ->

- Depthwise conv → Sigmoid, multiplied into skip
- Sharpen boundaries

SE (Squeeze-Excitation)

- Channel attention via global avg pool + MLP
- Reweight channels globally

Multi-scale pooled cross-attention

- Decoder queries skip at pool sizes (4, 8)
- Let decoder features guide which skip info to use

Learned gate

- Sigmoid over [skip, ctx] concat
- Control how much attention context to blend in

Dice = 0.9114, IoU = 0.8397 at threshold 0.30
The BSEI debug stats show the gate is ~0.5 (active), and the ctx/skip ratio is 0.55–0.83, meaning the attention context is meaningfully contributing.

CVC-300 -> 61
ETIS-LARIBPOLYPDB -> 196
CVC-COLONDB -> 380
CVC-ClinicDB -> 61 Test and 551 Train
Kvasir-SEG -> 100 Test and 900 Train
