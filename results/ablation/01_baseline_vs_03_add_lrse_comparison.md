# Ablation Comparison: Baseline vs. Baseline + LRSE/BSEI

## Summary

This report compares the completed three-seed results for:

- `01_baseline`
- `03_add_lrse` with MSC disabled

LRSE/BSEI does not improve upon the baseline. Mean mDice and mIoU decrease on all five datasets, while MAE increases on all five datasets.

## Primary metrics

| Dataset | Baseline mDice | +LRSE mDice | Delta mDice | Baseline mIoU | +LRSE mIoU | Delta mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG | 0.9171 | 0.9132 | -0.0040 | 0.8611 | 0.8558 | -0.0053 |
| CVC-ClinicDB | 0.9328 | 0.9230 | -0.0097 | 0.8792 | 0.8656 | -0.0136 |
| CVC-300 | 0.8894 | 0.8782 | -0.0113 | 0.8156 | 0.8029 | -0.0126 |
| CVC-ColonDB | 0.7758 | 0.7701 | -0.0057 | 0.6944 | 0.6851 | -0.0093 |
| ETIS-LaribPolypDB | 0.7707 | 0.7415 | -0.0292 | 0.6884 | 0.6597 | -0.0288 |

Mean mDice across the five datasets decreased from **0.8572** to **0.8452**, an absolute change of **-0.0120**.

## Paired-seed consistency

- CVC-300 and ETIS-LaribPolypDB were worse with LRSE for all three paired seeds.
- Kvasir-SEG, CVC-ClinicDB, and CVC-ColonDB were worse with LRSE for two of the three paired seeds.
- The largest mean degradation occurred on ETIS-LaribPolypDB at **-0.0292 mDice**.

## Complexity

| Measure | Baseline | +LRSE | Change |
|---|---:|---:|---:|
| Parameters | 29,190,098 | 29,197,010 | +6,912 (+0.024%) |
| GMACs | 13.6011 | 13.6395 | +0.0385 (+0.28%) |
| GFLOPs | 27.2021 | 27.2791 | +0.0770 (+0.28%) |

## Conclusion

LRSE/BSEI should not be promoted as a standalone addition. It is less harmful than MSC overall, but it still adds computation and reduces mean performance and external-dataset generalization. The baseline remains the best completed configuration.

The next useful experiment is the detail branch directly on the baseline, with both MSC and LRSE disabled.

These results are descriptive means and sample standard deviations across three seeds. They should not be treated as a definitive statistical-significance claim.

## Sources

- `results/ablation/01_baseline/summary.csv`
- `results/ablation/03_add_lrse/summary.csv`
