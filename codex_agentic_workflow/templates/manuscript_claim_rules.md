# Manuscript Claim Rules

1. Never invent a metric, percentage, dataset size, p-value or improvement.
2. Every numeric claim must reference an aggregated result file.
3. Report repeated-run results as `mean ± std`.
4. Explain whether the standard deviation is across seeds or images.
5. Do not claim a module improves Dice if the confidence interval and raw runs do not support it.
6. Keep validation-based checkpoint selection separate from final test reporting.
7. State the exact threshold policy and whether TTA was used.
8. Cite every external GitHub implementation and respect its license.
9. Replace placeholders only after the Reviewer QA Gate passes.
