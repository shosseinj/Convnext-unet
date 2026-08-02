---
description: Reconcile workflow requirements with the current Codex-modified repository
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

Inspect git status/diff, relevant source files, existing workflow files, tests, results, checkpoints, processes, locks, and logs. Produce CURRENT_STATE.md with: already implemented, partially implemented, missing, conflicting, active processes, valid evidence, and safest next action. Do not edit source code.

Read AGENTS.md and the workflow configs. Work from current repository evidence, not assumptions. Return a compact structured result with status, evidence paths, blockers, and next transition. Do not modify unrelated files.

