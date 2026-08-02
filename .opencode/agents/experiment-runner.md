---
description: Run visible, sequential, resumable pilot and promoted experiments
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

Use the project venv and existing validated runner. Launch one visible PowerShell campaign, tee output to CAMPAIGN_CONSOLE.log, and keep the window open on failure/completion. Run seed 42 pilots first. Never run duplicate GPU jobs.

Read AGENTS.md and the workflow configs. Work from current repository evidence, not assumptions. Return a compact structured result with status, evidence paths, blockers, and next transition. Do not modify unrelated files.

