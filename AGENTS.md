# ConvNeXt U-Net Agentic Workflow Rules

## Mission
Continue from the repository's current real state. Preserve valid work already performed by Codex or other agents. Audit ConvNeXt, implement and evaluate UGBR, and use evidence-based gates.

## Non-negotiable rules
- First reconcile the current repository; never assume the bundled workflow is newer than the code.
- Do not overwrite valid existing implementations, checkpoints, results, or reports.
- Stop duplicate/stale campaign processes before launching a new GPU run.
- Only one GPU training process may run at a time.
- Input resolution is 352x352.
- New comparison matrix: baseline, baseline_ugbr, baseline_best_existing, baseline_best_existing_ugbr.
- Resolve `baseline_best_existing` from verified existing three-seed results; expected candidate is `baseline_msc_bsei_detail`.
- Required loss: Lseg(final) + 0.4 Lseg(initial) + 0.2 Lboundary + 0.1 Lconsistency.
- ConvNeXt audit must PASS before new pilots.
- Run seed 42 pilots first. Promote to three seeds only when paired UGBR gain is >= 0.003 absolute Dice and no required safety metric materially degrades.
- Do not claim Dice > 0.94 in advance. Report measured results only.
- Do not modify manuscript/DOCX during this workflow.
- Keep user-facing status in RUN_STATUS.md, maximum 15 lines.
- Stream visible training output to CAMPAIGN_CONSOLE.log.
- Do not commit or push unless the user explicitly asks.

## Workflow references
Read these before action:
- agentic_workflow/configs/graph.yaml
- agentic_workflow/configs/loops.yaml
- agentic_workflow/configs/experiment_matrix.yaml
- agentic_workflow/docs/WORKFLOW.md
