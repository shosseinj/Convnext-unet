---
description: Audit ConvNeXt identity, weights, stages, preprocessing, optimizer, and gradients
mode: subagent
temperature: 0.1
maxSteps: 80
permission:
  read: allow
  glob: allow
  grep: allow
  list: allow
  bash: ask
  edit: deny
  task: deny
  websearch: allow
  webfetch: allow
---

Perform the ConvNeXt audit defined in the workflow. Prefer targeted executable checks. Create/update CONVNEXT_AUDIT.md under 40 lines. Do not implement UGBR. Audit result must be PASS or FAIL.

Read AGENTS.md and the workflow configs. Work from current repository evidence, not assumptions. Return a compact structured result with status, evidence paths, blockers, and next transition. Do not modify unrelated files.

