# OpenCode Multi-Agent Workflow

## Purpose
Reconcile work already completed by Codex, prove ConvNeXt correctness, complete any missing UGBR implementation, and evaluate it with fair gated experiments.

## Reconciliation contract
The installed workflow contains instructions, not a replacement project snapshot. The repo-reconciler must inspect current diffs and classify each requirement as implemented, partial, missing, conflicting, or unverifiable. Existing correct code wins. No source file is replaced merely because its design differs from an earlier prompt.

## ConvNeXt audit gate
Verify exact variant/library, pretrained source and SHA-256, matched/missing/unexpected keys, loaded coverage, RGB/range/normalization, 352x352 stage shapes, finite feature statistics, intended skip mapping, optimizer inclusion, freeze/unfreeze, discriminative learning rates, scheduler groups, nonzero encoder/decoder gradient norms, and actual encoder weight update after unfreeze. Produce CONVNEXT_AUDIT.md (<=40 lines), PASS or FAIL.

## UGBR contract
The model exposes initial_logits and final_logits. UGBR uses decoder context, shallow boundary evidence, and uncertainty derived from the initial prediction. The final output is residual refinement: final_logits = initial_logits + refinement_logits. Preserve correct existing Codex implementation when it satisfies this contract.

## Loss contract
Total = Lseg(final) + 0.4 Lseg(initial) + 0.2 Lboundary + 0.1 Lconsistency. Boundary targets come from masks without new annotations. Ensure no accidental duplicate boundary weighting.

## Experiment policy
Use identical split, preprocessing, augmentation policy, resolution, epoch budget, checkpoint rule, and evaluation protocol across paired variants. First run seed 42. Promote only with >=0.003 absolute Dice improvement for a paired UGBR comparison and no material degradation in required safety metrics. No guarantee or fabricated target of >0.94.

## Visible execution
Use the existing project venv. Launch one visible PowerShell process with unbuffered Python output and Tee-Object append to CAMPAIGN_CONSOLE.log. Validate each run before the next starts. Do not keep OpenCode polling a healthy job.

## User-facing files
RUN_STATUS.md is <=15 lines and current-only. CAMPAIGN_CONSOLE.log is the single visible campaign log. Internal artifacts remain in their existing directories.
