# Loop Engineering

## Technical repair loop

`failure -> fingerprint -> diagnose -> minimal patch -> targeted test -> independent validation`

حداکثر دو تلاش. تکرار همان fingerprint باعث BLOCKED می‌شود.

## Architecture learning loop

`pilot -> paired delta -> inspect failure modes -> one bounded UGBR revision -> repilot`

فقط یک revision مجاز است. split، metric، seed و test selection قابل تغییر نیستند.

## Experiment loop

`preflight -> train -> artifact check -> validator -> transition`

خروج process به معنی موفقیت نیست. validator باید PASS کند.

## Token-control loop

Codex فقط در eventها فعال می‌شود: launch، completion، failure، validation و transition. polling epoch ممنوع است.
