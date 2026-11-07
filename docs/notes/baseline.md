# The fixed-pipeline baseline (S2.3)

Spec: §19.3, §19.2. Code: `hypora/evaluation/baselines/fixed_pipeline.py`. Interface:
`hypora/evaluation/system.py` (`EvalSystem`). Tests: `tests/baseline/`,
`tests/unit/evaluation/test_fixed_pipeline.py`.

The claim it tests: the agent beats simple automation. So it has to be simple, and it has to be
compared fairly: same dataset, same budget, same harness, same sandbox, same evidence.

## What it does

| Choice | Value | Why |
|---|---|---|
| Split | `random`, 20% test, `kfold_3`, seed 1990 | Spec §19.3. Random even for classification, so it isn't stratified either |
| Columns | All but the target and `forbidden_columns` | A contract can't include a forbidden column; everything else stays, leak, group key, and time key included |
| Model | `templates/tabular/gradient_boosting/pipeline.py`, unchanged, empty config | Default HistGradientBoosting |
| Trials | One | No search, no retry |
| Primary metric | The case's; if unnamed, D-019's default for the task: `roc_auc` (binary), `f1_macro` (multiclass), `mae` (regression) | The same default the agent's Intake applies |
| Reported metrics | Every metric the platform computes for the task | So the honesty check has a stated value whichever metric a case grades when its constraints name none, including `brier_score` |
| Decision class | `train`, once its contract is approved; none if it stopped before that | It always trains, so it fails `reuse` and `infeasible` cases, as it should |
| Job limits | The configured limits, lowered to the case's `max_job_limits`; no GPU | `auto_within_contract` approves only inside the bounds |

It reads no request text, no trap, and no profile finding. It receives `SystemInput`, so it can't see
a trap or a hidden row at all. Exclusion rules and named models are plain language or a model choice,
which it doesn't act on, and a note says so.

## How it runs

`_BaselineRun` subclasses the controller's `_Run`, so `execute()`, the state table, `RunGuard`, the
private dataset copy, worker handling, draining, `select_final`, the final job, and the event log are
the agent's code, not a copy. Only the steps an LLM fills are replaced:

| Step | Agent | Baseline |
|---|---|---|
| Intake | Model drafts, code settles | Code, from the case's constraints, with Intake's own checks |
| Contract | `build_data_contract`, `build_plan` | Same builders where they apply; a random split; a one-measurement plan; the same `_approve` |
| Phase B | Model calls tools through MCP | Two fixed calls through the same MCP server: `write_trial_files`, `submit_experiment` |
| Waiting | `get_experiment_status` | The job records, as the controller does, so polling spends no steps |
| Report | Facts, then narrative | Facts, with a note that there is no narrative |

The job is validated, then the controller's `final` job re-validates, refits, scores the test split
once, and runs verifier v1, exactly as for the agent.

Going through MCP rather than calling the tool class means the trial and job pass the same checks,
and every step has `tool_call`, `policy_decision`, and `tool_result` events that S2.2's policy checks
read the same way for both systems.

**What the log looks like.** The MCP session's role is `single_agent`, the only role with an allowlist
(changing that is trigger 3). A baseline run is told apart by what's missing: no `model_call`, no
`role_started`, no agent handoff. "No LLM" holds by construction: the controller has no adapter, and
the run replaces every step that would use one. This also means an `eval` profile with unset model IDs
can't stop the baseline.

## Records the graders read

S2.2 fails closed on missing evidence, so a baseline run leaves all of this under its `run_id`: a `runs`
row and events (even when refused); the experiment contract, truthfully stating the random split and
the columns, which `prepare_data` can replay on the visible file; the `trials` row, kind `baseline`;
and `approval_requested`, `approval_granted`, and the `approvals` row for the contract's plan hash.

`FinalModel` comes from `report.json`, so the reported metrics are the numbers the report states. The
model and pipeline are the final job's, each re-hashed when the `SystemRun` is built. A metric the
harness left undefined has no value, so it isn't reported.

`completed` means a verified final model. If the verifier fails, `final_model` is still set (it is the
model the run finished with) and `completed` is false.

## Failures

A failure returns a `SystemRun` with `completed=False` and notes. Refused cases (non-tabular family, or
a task the harness doesn't score) run no pipeline but still record a run, an `error` event, the
`INTAKE -> REPORTING -> FAILED` transitions, and a report. Only when the state database itself can't be
opened is there nothing to record. A programming error inside the run becomes a `CONTROLLER_ERROR`
event naming only its type, as in the controller, so it fails the run rather than the whole evaluation.

## Budget

`RunGuard` checks every tool call and the wait for each job, so trials, steps, wall clock, compute, and
user cancel stop it the way they stop the agent. It shares the guard, so `max_usd: 0` or
`max_llm_tokens: 0` also stops it, even though it spends neither.

## Injected faults

No platform code injects faults yet, so the baseline applies what it can and says what it can't:

- A `tool_error` on `write_trial_files` or `submit_experiment`, first call: the tool body raises a
  structured error through the same MCP boundary, with its allowlist, argument checks, result
  firewall, and call/policy/result events. No write or job is performed. An `error` event also marks
  the injected fault, and the pipeline stops. A fixed pipeline has no recovery.
- Any other `tool_error`: it is never reached, and a note says so.
- `malformed_data`: already in the dataset.
- Every other kind (OOM, timeout, crash): a note says it wasn't applied, so the run can't look like it
  survived a fault it never met.

## Known limits

- Every supported metric is required, including `brier_score` for binary tasks. The current harness
  raises when a scored fold cannot produce Brier (for example, a fold missing a class). Such a job
  fails explicitly and the baseline reports its error; it never silently omits a grading metric.
- A random split can leave a rare class out of a training fold, which the harness refuses
  (`labels outside the training class vocabulary`). The run then fails with that job's code. That is
  what a naive split costs.
- One-hot encoding in the template is dense, so a high-cardinality text column can reach the memory
  limit. The run reports `OOM_KILLED`, and the case's limits apply.
