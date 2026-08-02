---
description: Implement and verify the composite UGBR loss
mode: subagent
temperature: 0.1
maxSteps: 80
permission:
  read: allow
  glob: allow
  grep: allow
  list: allow
  bash: ask
  edit: ask
  task: allow
  websearch: allow
  webfetch: allow
---

Reconcile existing loss code. Ensure exact weighted terms, finite gradients, correct boundary target generation, and consistency masking. Avoid duplicate boundary weighting. Add focused tests only.

Read AGENTS.md and the workflow configs. Work from current repository evidence, not assumptions. Return a compact structured result with status, evidence paths, blockers, and next transition. Do not modify unrelated files.

