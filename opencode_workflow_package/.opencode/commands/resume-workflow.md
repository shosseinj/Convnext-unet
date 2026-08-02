---
description: Resume the saved research workflow from its real persisted state
agent: build
subtask: false
---

Resume the repository workflow from its real persisted state.

Before doing anything:
1. Read `codex_agentic_workflow/AGENTIC_WORKFLOW.md`, `.agentic/state.json`, and `RUN_STATUS.md` when present.
2. Inspect real matching processes, GPU state, active history/checkpoint, and validators.
3. If a healthy workflow/training process already exists, do not launch another one. Report the current state and stop.
4. If stale processes or locks exist, do not guess. Use `/stop-workflow` only when stopping is required and explicitly justified.
5. Continue only the earliest unmet workflow gate.
6. Long training must run in one visible PowerShell terminal with unbuffered output and append to `CAMPAIGN_CONSOLE.log`.
7. Do not remain open polling a healthy long-running job.
8. Keep `RUN_STATUS.md` concise and user-facing.
9. Never overwrite valid completed or legacy experiment artifacts.
10. Do not advance a gate until its validator passes.

Return a concise result containing the current stage, action performed, active process, visible terminal status, log path, error, and next action.
