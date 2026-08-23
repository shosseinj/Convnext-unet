# Ablation Comparison: Baseline vs. Baseline + Detail Branch

## Summary

This report compares the completed three-seed results for:

- `01_baseline`
- `04_add_db` with MSC and LRSE/BSEI disabled

The standalone detail branch does not improve upon the baseline. Mean mDice and mIoU decrease on all five datasets, while MAE increases on all five datasets.

## Primary metrics

| Dataset | Baseline mDice | +Detail Branch mDice | Delta mDice | Baseline mIoU | +Detail Branch mIoU | Delta mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG | 0.9171 | 0.9134 | -0.0038 | 0.8611 | 0.8551 | -0.0060 |
| CVC-ClinicDB | 0.9328 | 0.9276 | -0.0052 | 0.8792 | 0.8712 | -0.0080 |
| CVC-300 | 0.8894 | 0.8847 | -0.0048 | 0.8156 | 0.8099 | -0.0057 |
| CVC-ColonDB | 0.7758 | 0.7671 | -0.0087 | 0.6944 | 0.6836 | -0.0108 |
| ETIS-LaribPolypDB | 0.7707 | 0.7399 | -0.0308 | 0.6884 | 0.6588 | -0.0297 |

Mean mDice across the five datasets decreased from **0.8572** to **0.8465**, an absolute change of **-0.0106**.

## Paired-seed consistency

- CVC-ColonDB and ETIS-LaribPolypDB were worse with the detail branch for all three paired seeds.
- Kvasir-SEG, CVC-ClinicDB, and CVC-300 were worse for two of the three paired seeds.
- The largest mean degradation occurred on ETIS-LaribPolypDB at **-0.0308 mDice**.

## Complexity

| Measure | Baseline | +Detail Branch | Change |
|---|---:|---:|---:|
| Parameters | 29,190,098 | 29,215,154 | +25,056 (+0.086%) |
| GMACs | 13.6011 | 14.3693 | +0.7682 (+5.65%) |
| GFLOPs | 27.2021 | 28.7385 | +1.5364 (+5.65%) |

## Conclusion

The standalone detail branch should not be promoted. It is the least harmful of the tested additions so far, but it still reduces mean segmentation performance and external-dataset generalization while increasing computation substantially. The baseline remains the best completed configuration.

These results are descriptive means and sample standard deviations across three seeds. They should not be treated as a definitive statistical-significance claim.

## Sources

- `results/ablation/01_baseline/summary.csv`
- `results/ablation/04_add_db/summary.csv`
