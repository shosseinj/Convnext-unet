---
description: Show the real workflow and GPU status without changing anything
agent: build
subtask: false
---

Inspect the real current workflow state without changing files or processes.

Check only:
- `RUN_STATUS.md`
- matching repository processes for campaign, queue, training, and resume watcher
- the active run history/checkpoint timestamp
- `nvidia-smi` compute processes when available

Report no more than 10 lines:
Stage, active run, latest epoch, completed count, process state, GPU state, last artifact update, error, next action.
Do not launch, stop, resume, patch, or continuously monitor anything.
