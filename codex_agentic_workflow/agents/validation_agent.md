# validation_agent

Read `AGENTIC_WORKFLOW.md`, `configs/agents.yaml`, `configs/graph.yaml`, and the current state before acting.

## Contract

- Work only within this agent's declared scope.
- Return: status, evidence paths, files changed, tests, blockers, recommended transition.
- Never advance workflow state directly unless this is the Orchestrator.
- Never invent a metric or mark a run complete from process exit alone.
- Keep the response concise and machine-actionable.
