---
description: Resume the saved ConvNeXt experiment workflow
---

Resume the project workflow from the real persisted state.

Repository:
C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet

Use only this Python executable for tests, validators, launchers, and training:

C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe

Never use the global Python installation.

Required behavior:

1. Read:
   - AGENTS.md
   - .agentic/state.json
   - RUN_STATUS.md, if present
   - the current workflow configuration
   - relevant validators and launchers

2. Reconcile saved state with reality:
   - active repository processes
   - GPU processes
   - lock files
   - checkpoints
   - histories
   - summaries
   - validation outputs

3. Never trust state.json alone.

4. If a valid training or campaign process is already active:
   - do not start another process
   - report the real active run
   - update RUN_STATUS.md
   - stop the OpenCode session

5. If no process is active:
   - identify the earliest unmet workflow gate
   - validate all existing completed artifacts
   - preserve official, legacy, pilot, and partial outputs
   - never overwrite or reuse incompatible result directories

6. For PILOT_SEED_42:
   - use the canonical four-variant pilot matrix
   - use seed 42
   - use a fresh isolated pilot namespace
   - reject output paths that overlap legacy or official results
   - run only one GPU job at a time
   - validate each pilot before starting the next

7. Run every long training campaign in a visible PowerShell terminal.

8. The visible terminal must:
   - use unbuffered Python output
   - show stdout and stderr live
   - append the same output to CAMPAIGN_CONSOLE.log
   - remain open after completion or failure

9. Update only this concise user-facing file:
   RUN_STATUS.md

10. RUN_STATUS.md must remain under 15 lines and contain only:
    Stage:
    Campaign status:
    Active run:
    Latest epoch:
    Completed runs:
    Last completed run:
    Last result:
    Process status:
    Last error:
    Next action:
    Console log:
    Last update:

11. After each run exits:
    - inspect the relevant log section
    - inspect required artifacts
    - run the existing validator
    - continue only after validation PASS
    - stop on failure

12. Do not continuously monitor healthy training inside OpenCode.

13. Do not work on manuscript, Word, controls, qualitative evaluation, or statistical analysis before their workflow gate.

14. Do not create extra status reports.

Final response must be brief and contain only:
Stage:
Status:
Active run:
Completed:
Visible terminal:
GPU:
Console log:
Status file:
Error:
Next:
