# OpenCode workflow controls

Use these project commands:

- `/workflow-status`: inspect real process, GPU, and artifact state without changing anything.
- `/stop-workflow`: invoke the dedicated Stop-Job Agent and safely stop this repository's active workflow.
- `/resume-workflow`: continue the earliest unmet gate from persisted state.

Never run `/resume-workflow` while a healthy training process is already active. Use `/workflow-status` first.
