# Waiting on long jobs (D-049)

`get_experiment_status` already waited on the server (D-030), but at most 45 s per call. Each call
is one model turn that resends the conversation, so a 30-minute job cost about 40 turns of pure
waiting, against a 60-turn role cap, the step budget and the token budget. PyTorch training is
planned, so jobs of 30 minutes and more must be affordable.

## Design

The tool and its single `job_id` argument are unchanged. Only the server-side bound of one call
changes (`hypora/mcp_server/tools.py`, `status_wait_bound`):

```
bound(n) = min(first * 2**n, cap, job.limits.max_wall_seconds)      (first = 0 means no wait)
```

`n` counts the earlier calls on the same job, in this MCP session, that sat out their whole bound
with the job in the same phase (`LEASED` and `RUNNING` are one phase, as before). Any change resets
`n`, and a terminal state returns at once. A cancelled wait leaves `n` as it was. With the defaults
(45 s, cap 600 s) the bounds are 45, 90, 180, 360, 600, 600, ... s, so a model that calls again at
once makes:

| Job runs | Calls before (45 s fixed) | Calls now |
|---|---|---|
| 1 min | 2 | 2 |
| 10 min | 14 | 4 |
| 30 min | 40 | 6 |
| 2 h | 160 | 15 |

Counting quiet waits, rather than reading a job start time, needs no new job field or schema. The
cap still wakes the model every 10 minutes at most, so it can notice a stuck job and cancel it.
The per-stage limit bound keeps a wait from outlasting a stage that should already have ended.

## Everything on the path, checked

| Limit | Finding |
|---|---|
| MCP client read timeout | 300 s (`MCP_READ_TIMEOUT_S`) would cut a 600 s wait. Waiting tools now get `WAITING_CALL_READ_TIMEOUT_S` (720 s) per call in `McpToolChannel`; other tools keep 300 s. |
| Session settings | `status_wait_seconds` (first, default 45 s) and new `status_wait_cap_seconds` (default 600 s), both validated to at most `MAX_STATUS_WAIT_SECONDS` (600 s). Written only by the trusted launcher. |
| MCP Inspector | The dev session sets both to 5 s, under the Inspector's 10 s request timeout. |
| Transport keep-alive | stdio only: no idle timeout or keep-alive exists to expire. |
| User cancel, wall clock, compute, spend, tokens, steps | `_call_watching_limits` asks the guard every second during a waiting call and cancels it, and the server ends the wait on cancellation (D-030). Covered with a 10-minute wait in force (`tests/controller/test_status_wait.py`). |
| Role turn cap (60) and step budget | Fewer status calls spend fewer turns and steps. |
| C worker time budget | `worker_scope` cancels a worker at its brief's timeout (default 300 s), wait included. A worker that must wait on a long job needs a larger brief timeout; that is C's budget choice and is not changed here. |
| Post-agent drain and final job (D-047) | They poll the queue directly, bounded by the job limits and the run's remaining wall clock. They do not use the status tool and are unchanged. |
| Events | A waiting call still records one `tool_call` and one `tool_result` (with its latency, or `CALL_CANCELLED`). The worker records job state changes during the wait. No new event type. |
| Run budgets (`configs/budgets.yaml`) | Run wall clock is 7,200 s and sandbox compute 3,600 s. A 30-minute job uses half the compute budget; longer PyTorch runs will need larger budgets, a separate decision. |
| Scripted test doubles | Unchanged: the scripted model calls again at once. Tests that need short fixed waits set both settings (`_short_waits`). The schedule itself is tested on a fake clock, without sleeping. |

## Model-call timeout

`retry.request_timeout_s` is now 240 s (it was 120 s; the slowest real call took 73 s). It is a
per-attempt HTTP timeout shared by every profile. With 3 retries and a 6 s backoff base, one
logical model call can take at most about 4 x 240 + 42 s if a provider hangs every time. Run
limits are checked before each model call, not during one, so a user cancel issued during a hung
call is acted on when that call ends, as before.
