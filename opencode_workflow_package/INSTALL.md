# OpenCode Workflow Overlay

This package adds three OpenCode commands to an existing project:

- `/workflow-status` — read-only real status check
- `/stop-workflow` — safely stop only this project's campaign/queue/training/watcher processes
- `/resume-workflow` — continue from the persisted workflow state

It also adds the dedicated `stop-job` subagent and `tools/stop_workflow.ps1`.

## Install on Windows

1. Extract this ZIP.
2. Open PowerShell in the extracted folder.
3. Run:

```powershell
.\install.ps1 -ProjectRoot 'C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet'
```

4. Close and reopen OpenCode from the project root:

```powershell
cd C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet
opencode
```

5. Type:

```text
/workflow-status
```

To stop the current workflow:

```text
/stop-workflow
```

To continue later:

```text
/resume-workflow
```

## Safety behavior

`/stop-workflow` matches both the repository path and known workflow script names. It does not delete results, checkpoints, histories, metadata, or logs. Locks are removed only after matching processes are gone.

## Manual fallback

From the project root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\stop_workflow.ps1
```
