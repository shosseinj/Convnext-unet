# Ablation Comparison: Baseline vs. Baseline + MSC

## Summary

This report compares the completed three-seed results for:

- `01_baseline`
- `02_add_msc`

The baseline performs better on every dataset and every reported accuracy metric. MSC increases model size and computation while reducing both in-domain and external-dataset performance.

## Primary metrics

| Dataset | Baseline mDice | +MSC mDice | Delta mDice | Baseline mIoU | +MSC mIoU | Delta mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG | 0.9171 | 0.9100 | -0.0072 | 0.8611 | 0.8512 | -0.0100 |
| CVC-ClinicDB | 0.9328 | 0.9217 | -0.0110 | 0.8792 | 0.8632 | -0.0160 |
| CVC-300 | 0.8894 | 0.8700 | -0.0195 | 0.8156 | 0.7928 | -0.0227 |
| CVC-ColonDB | 0.7758 | 0.7686 | -0.0072 | 0.6944 | 0.6811 | -0.0133 |
| ETIS-LaribPolypDB | 0.7707 | 0.7379 | -0.0328 | 0.6884 | 0.6571 | -0.0314 |

Mean mDice across the five datasets decreased from **0.8572** to **0.8416**, an absolute change of **-0.0155**.

## Other metrics

MSC also produced worse mean `F_beta_w`, `S_alpha`, `mE_phi`, `maxE_phi`, and MAE on all five datasets. The largest mDice degradation occurred on ETIS-LaribPolypDB at **-0.0328**.

Across paired seeds, MSC was worse for all three seeds on CVC-300 and ETIS-LaribPolypDB. On Kvasir-SEG, CVC-ClinicDB, and CVC-ColonDB, it was worse on two of the three seeds.

## Complexity

| Measure | Baseline | +MSC | Change |
|---|---:|---:|---:|
| Parameters | 29,190,098 | 29,572,851 | +382,753 (+1.31%) |
| GMACs | 13.6011 | 13.6460 | +0.0449 (+0.33%) |
| GFLOPs | 27.2021 | 27.2920 | +0.0899 (+0.33%) |

## Conclusion

MSC alone is not beneficial under this ablation protocol. It adds parameters and computation while decreasing segmentation performance and external-dataset generalization. The baseline should remain preferred unless a later component demonstrates a positive interaction with MSC.

These results are descriptive means and sample standard deviations across three seeds. With only three seeds, they should not be treated as a definitive statistical-significance claim.

## Sources

- `results/ablation/01_baseline/summary.csv`
- `results/ablation/02_add_msc/summary.csv`
