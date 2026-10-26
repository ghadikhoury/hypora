---
prompt: single_agent_context
version: 1
---
Run ID: $run_id (idempotency keys start with "$run_id:")

## Task
$task

## Experiment contract (approved; fixed)
$contract

## Dataset profile (aggregates only)
$profile

## Limits for this run
$limits

## A working pipeline.py to start from (logistic or linear regression)
```python
$example
```

Begin with `inspect_dataset`.
