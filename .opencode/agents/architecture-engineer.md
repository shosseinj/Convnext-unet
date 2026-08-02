---
description: Implement or reconcile UGBR and model variants with minimal code changes
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

After audit PASS, inspect any Codex implementation first. Keep correct existing code. Implement only missing or incorrect UGBR behavior: initial logits, uncertainty-guided boundary refinement, residual final logits, and four canonical variants. Add focused tests.

Read AGENTS.md and the workflow configs. Work from current repository evidence, not assumptions. Return a compact structured result with status, evidence paths, blockers, and next transition. Do not modify unrelated files.

