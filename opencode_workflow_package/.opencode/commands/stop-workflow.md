---
description: Stop this repository's running workflow safely
agent: stop-job
subtask: true
---

Stop the currently running experiment workflow for this repository now.

Use the dedicated stop script, preserve all results and checkpoints, remove only stale workflow locks after shutdown, update `RUN_STATUS.md`, and verify that no matching campaign, queue, training, or resume-watcher process remains.
