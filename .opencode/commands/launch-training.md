---
description: Launch and verify the approved training campaign
agent: training-launcher
---

---

Launch the currently approved workflow training action.

Do not perform general workflow analysis.

Read the persisted stage and launch only the campaign already approved by the workflow.

A launch is successful only when:

- A visible PowerShell window exists.
- Its process remains active.
- The approved queue process exists.
- The console log advances.
- The expected run starts or resumes.
- GPU activity is verified when training begins.
- No duplicate job exists.

If launch verification fails, record START_FAILED and report the exact error. Do not claim that training was launched.
