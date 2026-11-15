# Recorded readiness and configuration lock (S4.2, D-051)

The single new full development readiness collection (D-036). It covers the six dev cases on the
pinned `eval` profile, for the multi-agent A/B/C system (`configs/eval/default.yaml`) and the
single-agent comparison point (`configs/eval/single_agent.yaml`), each against the fixed baseline.
The six dev cases are development controls, not held-out test results (S5).

## Procedure

Every arm ran from the clean commit `1fa719d43a9f7262587ac8c7441f6179cf4abae6` (S4.2 code on
`main` 5ffc02f), one run at a time under `flock /tmp/hypora-heavy.lock`, through the normal runner
(same controller, MCP server, worker, sandbox, hidden splits and programmatic graders as `make eval`):

```bash
uv run --locked python -m scripts.readiness_eval run --config default --systems baseline
uv run --locked python -m scripts.readiness_eval run --config default --resume          # agent
uv run --locked python -m scripts.readiness_eval run --config single_agent --systems baseline
uv run --locked python -m scripts.readiness_eval run --config single_agent --resume     # agent
uv run --locked python -m scripts.readiness_eval lock                                   # lock.json
```

The collector reuses the S2.6 durable paid ledger: each real OpenAI SDK attempt, retries included,
reserves $0.36 (Luna's absolute per-request bound) before dispatch, and only provider-reported
usage settles a reservation. The journal is ignored under `runs/readiness/`, and the public audit is
`results/readiness/paid-accounting.json` (IDs, counts and prices only).

Caps were enforced by code:
- $10 in aggregate.
- No comparison starts if spent plus reserved plus its full case budgets could pass $10.
- At most $5 per comparison at full budgets (`--max-total-usd 5`).
- $0.60 per case run (the merged case budget).

The approval policy was the configs' `auto_within_contract`, so no interactive gate appeared.
`leakage_flag` was `on`, the merged default and the full system as S4.1b defines it.

**OpenAI balance.** It was not read. The key lacks the `api.usage.read` scope, and the
credit-grant endpoint needs a browser session. `GET /v1/models/gpt-6-luna` returned 200, and one
probe call (13 input and 5 output tokens, $0.0000038, served `gpt-6-luna`) confirmed the account
had quota. The probe is counted in the ledger. No quota or billing error occurred.

## Measured estimate (before the first paid call)

The estimate used the recorded `resources.usd` of the five eval-profile A/B/C live runs on
credit-g (2026-11-14) that reached training: `run_7a9185d5f5ce` $0.0456, `run_9a6029d4c1be`
$0.0529, `run_b2d5f594bf4b` $0.0470, `run_6cd1967c79cf` $0.0438 and `run_6e893a2f8062` $0.0396.
The mean is $0.046 and the maximum $0.053. On dev, larger datasets used up to 1.9x the tokens.

- Expected total: about $0.6 (multi $0.30–0.55, single $0.05–0.35, baseline $0).
- High case, with all 12 runs at twice the maximum: $1.27.
- Hard bound: 12 × $0.60 = $7.20 at case budgets.

## Cost ledger (cap $10)

| Item | Attempts | Actual $ | Wall time |
|---|---:|---:|---:|
| Quota probe | 1 | 0.0000 | — |
| Baseline, `default` (no model) | 0 | 0.0000 | 543 s |
| Agent, `default` (multi), six cases | 142 | 0.1065 | 1,950 s |
| Baseline, `single_agent` (no model) | 0 | 0.0000 | 553 s |
| Agent, `single_agent`, six cases | 60 | 0.0329 | 680 s |
| **Total** | **203** | **0.1394** | 3,726 s |

There were no retries, no infrastructure reruns, no fallback calls, no unknown reservations and
no prior attempts replaced. Every model call was `openai/gpt-6-luna` on the `eval` profile.

## Identity

| Field | Value |
|---|---|
| Commit | `1fa719d43a9f7262587ac8c7441f6179cf4abae6` |
| Run config hash, `default` | `63286a1d7cc9bffadf1bfa7eb6a6d5a53b7fbc6edc8474fff7e13415e3d4ebd8` |
| Run config hash, `single_agent` | `8efbe27b123dda6e13bf21920609a70c25ab5b831dc7854bc13bbe34443a9d29` |
| Tool contract SHA-256 | `1518dfe40ff96fbe0c7eb1dc94f8df685d2c706e17355910899b58d0ccea12e9` |
| Sandbox image | `hypora-sandbox-tabular:f50772e217853fa1` |
| Python | 3.12.14 |
| Model | `openai/gpt-6-luna`, every tier, temperature not configurable, no fallback |
| Retry | 3 retries, 6 s base, 60 s max, 240 s request timeout |
| Prices | `configs/prices.yaml`, Luna list price read 2026-11-07 |
| Run date | 2026-11-15, 07:03–08:06 UTC |

## Results

`summary.md`, `summary.json` and `regressions.json` are under
`results/1fa719d43a9f7262587ac8c7441f6179cf4abae6/{default,single_agent}/`. These are derived
from the 24 per-case records, which are not committed yet.

**Per-case records held back (open for Ghadi).** gitleaks' generic-api-key rule flags the
`split_key` line in 20 of the 24 records. Each of these lines is a deterministic 32-hex split
identifier, the same false-positive class as D-035. All 12 values are identical to the ones
already committed at S2.6 under D-035's exception. That exception is scoped to the S2.6 result
paths, and widening it is a change to the scanner configuration, which this step may not make.
The records are preserved byte-for-byte, outside git, in the readiness clone and in
`~/hypora-readiness-records-1fa719d.tar.gz` on the collection machine. If Ghadi approves
extending D-035 to `results/1fa719d43a9f7262587ac8c7441f6179cf4abae6/(default|single_agent)/(agent|baseline)/dev_*.json`
(same rule, same `split_key` line regex), a follow-up commit adds the exception and the records
unchanged. The tables below are read from those records.

| Case | Multi agent | Single agent | Baseline |
|---|---|---|---|
| `dev_class_imbalance` | fail: stopped at CONTRACT | fail: stopped at CONTRACT | pass (macro-F1 0.819) |
| `dev_clean_control` | **pass** (MAE 0.627) | fail: stopped at CONTRACT | pass (MAE 0.681) |
| `dev_grouped_rows` | fail: stopped at CONTRACT | fail: stopped at CONTRACT | fail (group overlap, MAE 11.18) |
| `dev_target_leakage` | fail: stopped at CONTRACT | fail: stopped at CONTRACT | fail (leak kept, ROC-AUC 0.960) |
| `dev_temporal_order` | **pass** (MAE 84.0) | fail: stopped at CONTRACT | fail (random split, future counts) |
| `dev_tool_failure` | **pass** (ROC-AUC 0.751, recovered) | **pass** (ROC-AUC 0.751, recovered) | fail (no retry) |

| Config | Agent | 95% Wilson | Baseline | 95% Wilson | Paired difference, 95% bootstrap CI |
|---|---|---|---|---|---|
| `default` (multi) | 3/6 | [0.188, 0.812] | 2/6 | [0.097, 0.700] | +0.167, [−0.333, +0.667]: inconclusive |
| `single_agent` | 1/6 | [0.030, 0.564] | 2/6 | [0.097, 0.700] | −0.167, [−0.667, +0.333]: inconclusive |

**No advantage over the baseline is claimed.** Both intervals include zero, and six cases cannot
separate these systems.

Per-run cost, time and steps (model calls plus tool calls):

| Case | Multi run | $ | Steps | Seconds | Single run | $ | Steps | Seconds |
|---|---|---:|---:|---:|---|---:|---:|---:|
| imbalance | `run_6f2f182ed669` | 0.0125 | 12 | 178 | `run_40f9bee52f36` | 0.0033 | 5 | 54 |
| clean control | `run_c672b0a47f54` | 0.0167 | 49 | 444 | `run_bd6ce6725b3d` | 0.0024 | 5 | 41 |
| grouped | `run_018aa9a34822` | 0.0108 | 13 | 140 | `run_e4f7d9ab51f7` | 0.0037 | 7 | 52 |
| leakage | `run_5c46b5d2cf35` | 0.0106 | 11 | 140 | `run_1723b4120fca` | 0.0030 | 5 | 41 |
| temporal | `run_c09636637da3` | 0.0280 | 61 | 608 | `run_e7689f6b7190` | 0.0038 | 7 | 56 |
| tool failure | `run_3eead42a66a7` | 0.0279 | 54 | 575 | `run_1e9e3c29e799` | 0.0167 | 55 | 473 |

Agent latency: multi mean 347.5 s and p95 607.7 s; single mean 119.6 s and p95 473.2 s
(nearest-rank). No run reached its case's step, token, compute or wall-clock limit (at most 61 of
100 steps). No run started a C worker.

**D-007 cost per passed trap (Luna only).** Multi passed one planted trap (temporal) for $0.1065,
about $0.107 per passed trap. Single passed none. Luna passed neither the leakage nor the grouped
trap, so by D-007's own rule no measured model is eligible (D-051).

## The main failure: C's open questions stop the run

Eight of the 12 agent runs (multi: imbalance, grouped, leakage; single: all except tool failure)
ended in `CONTRACT failed: MethodologyError: unresolved methodology questions require escalation`.
All of these are graded failures with complete records, not infrastructure faults.

C's data-contract proposal (`agent_c_methodology@v3`, used in both modes) still listed
`open_questions` after its one bounded correction. The platform validator
(`hypora/core/controller/methodology.py`) treats any remaining question as one only the user can
answer, and stops before training. An evaluation run has no user, and no clarification path exists
for methodology questions, so the run fails with no model.

The question text cannot be quoted: model outputs of this step are filtered before storage (D-038
addendum), and the refused proposal is never stored. The questions clustered on the trap datasets,
where a careful agent may want to ask whether a column is available at prediction time or how rows
are grouped. Earlier live `hypora run` runs on the untrapped credit-g goal never raised one.

This is product behavior, recorded as measured. Changing it would change behavior and need the
affected evidence rerun, so it was not done here (Open for Ghadi).

Other observations:
- **Research in eval.** The dev cases' merged `permitted_tools` do not include B's research tools,
  so B reported invalid or unsupported citations in every multi run. As a result A recorded an
  `unknown` expected score range. The S4.1b check therefore never investigated: four `leakage_check`
  events, all `trigger` with outcome `unknown_range`. The leakage flag is wired but unexercised by
  this evidence.
- **Temporal.** The multi agent excluded the future count columns, split by time and met the bar
  (MAE 84.0 ≤ 146). Neither S2.6 configuration did this.
- **Tool failure.** Both agents retried the injected `TOOL_UNAVAILABLE` and finished. The fixed
  pipeline cannot retry.

## Regressions and what can be compared

The runner found no comparable base: S2.6's records live under the squashed commit `ec81155`,
which is not an ancestor of `main`, and the `default` config has never been recorded. So
`regressions.json` reports `no_baseline`. A tool-level comparison would refuse anyway, because the
run-config hashes (`01ac3233…` at S2.6), case hashes for three cases, the tool contract (`70ca5ce7…`
to `1518dfe4…`), prompts, retry policy and the whole controller graph (Stage 1 single contract →
A/B/C) changed.

Case by case against the S2.6 `single_agent` eval records (same model and cases, different system):

| Case | S2.6 single | S4.2 single | S4.2 multi | Explanation |
|---|---|---|---|---|
| imbalance | pass | fail | fail | Regression: CONTRACT stop on C's open questions |
| clean control | pass | fail | pass | Single regression: CONTRACT stop; multi unchanged |
| grouped | fail (overlap) | fail | fail | Still failing, now at CONTRACT instead of the trap |
| leakage | fail (leak kept) | fail | fail | Still failing, now at CONTRACT instead of the trap |
| temporal | fail (trap) | fail | **pass** | Multi improvement: temporal split, future counts excluded |
| tool failure | pass | pass | pass | Unchanged |

The baseline reproduced S2.6 exactly: the same 2/6 and the same hidden metrics.

**`dev_target_leakage` grading question (open, not decided here).** Should this case require the
column to be "excluded" (strict, as merged) or accept "excluded or flagged"? It was graded with
the merged rule. In this run no agent reached a final model on this case, so the answer would not
have changed any grade.

## Locked configuration

`results/readiness/lock.json` locks:
- the eval models and retry policy;
- both configs' flags;
- `configs/budgets.yaml`;
- the version and SHA-256 of every runtime prompt (`agent_a_decide@v4`, `agent_a_intake@v1`,
  `agent_a_report@v2`, `agent_b@v2`, `agent_c@v5`, `agent_c_methodology@v3`, `agent_c_workers@v1`,
  `c_worker@v1`, `c_worker_final@v1`, `intake@v1`, `reporter@v1`, `role_plan@v1`,
  `single_agent@v3`, `single_agent_context@v1`, `single_agent_final@v1`);
- the tool contract hash.

`tests/unit/test_readiness_eval.py` fails when any of them changes. Regenerate the lock only
together with new evidence, or with a stated reason that the change cannot affect behavior.

### Limits and their source

Observed maxima across 14 live A/B/C runs that reached training, 2026-11-14/15, from
`~/hypora/runs`, `~/hypora-sweep/runs` and `~/hypora-s4.1b/runs`:

| Limit | Locked value | Observed max (run) | Headroom |
|---|---:|---:|---:|
| `max_c_workers` | 2 | 0 (no live run has started a worker) | n/a |
| `run.max_usd` | $2.00 | $0.107 list price (`run_c8e6591e07f9`, dev); Luna $0.053 (`run_9a6029d4c1be`) | ≥ 18x |
| `run.max_llm_tokens` | 1,000,000 | 664,263 (`run_03e1428ea7bd`); readiness single max 411,100 | 1.5x |
| `run.max_steps` | 200 | 118 (`run_b2d5f594bf4b`: 44 calls + 74 tools) | 1.7x |
| `run.max_trials` | 12 | 4 | 3x |
| `run.max_compute_seconds` | 3,600 | 427 (`run_03e1428ea7bd`) | 8.4x |
| `run.max_wall_clock_seconds` | 7,200 | 856 (`run_6e893a2f8062`) | 8.4x |
| `job_limits.max_wall_seconds` | 1,800 | 70 s per job (`run_96aacaacb07b`) | kept for D-049's long jobs |
| `job_limits.max_memory_gb` | 4.0 | 0.26 GB peak RSS (`run_c8e6591e07f9`) | kept |
| `job_limits.cpu_threads` | 2 | not a usage measure | kept |

The S1.7 Stage 1 single-agent runs that reached 1M tokens used a contract that no longer exists
(`docs/notes/cost-estimate.md`). The readiness runs stayed below every value (eval cases use their
own merged budgets). Nothing was
loosened. The job limits were not tightened to 1.5x of observed use, because 30-minute jobs are
planned (D-049).

## Unsupported behaviors (scoped out of the readiness claim)

- **Methodology clarification.** Any open question in C's data contract stops the run. No question
  is put to the user and the run does not continue on assumptions.
- **C workers** are implemented and tested with scripted models but have never run live.
- **The too-good-score investigation** (S4.1b) has never investigated in recorded evidence, because
  every expected range was `unknown` here.
- **B's research** is unavailable inside the dev cases, which do not permit research tools.
- **Families other than tabular** (vision, time series), experimental mode, GPU jobs, inference
  optimization, durable restart and replay, web/API, adaptive breadth and parallelism ablations
  (D-028, PLAN Later track). `durable_mode`, `breadth_mode` and the legacy worker names are inactive.
- **Inputs refused before any model call:** hard exclusion rules, named-model mandates, and resource
  or optimization constraints (`docs/notes/eval-runner.md`).
- **Malformed-data recovery** is ungraded; only a causal safe stop earns credit.
- **Static leakage analysis** of `fit`/`predict` is skipped (a reported limitation, D-044).

## Limitations

- Six development cases, one primary run each: wide intervals, and nothing about held-out
  performance. The repeat sample and test set belong to S5.
- One model. D-007's comparison was not run (D-051), and no other candidate has eligibility or cost
  evidence.
- The account balance was not read; a probe call stood in for it.
- The open-question text is not inspectable from stored records, so the reason for each stop is
  known only at the rule level.
- Timing is for one WSL2 machine (24 cores, 15 GB), with runs one at a time.
