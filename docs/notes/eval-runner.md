# Evaluation runner (S2.4)

`make eval CASES=dev SYSTEMS=agent,baseline CONFIG=single_agent PROFILE=dev` resolves one case
set, a named config under `configs/eval/`, and its model profile. `PROFILE` is optional: without
it, the named config chooses. `CASES` also accepts a directory. Every comparison uses this runner;
feature behavior comes from the existing flags in the named config. HYPORA environment overrides
are refused so a leftover shell setting cannot change a recorded comparison.

Cases run up to `max_parallel_cases` in the eval config (default 4), lowered by CPU/memory pool
capacity and optional `ARGS="--parallel 1"`. Model calls share provider admission across cases.
Each run has its own database and run ID. The splitter runs once per case, so both systems use
the same visible data. Systems receive an explicit SystemInput without traps, grading, or hidden
split settings. Hidden rows stay outside the repository and the run mounts.

## Records and resuming

A clean tree writes one primary record to
`results/{commit_sha}/{config}/{system}/{case_id}.json`. A profile override uses `{config}.{profile}`
for the config directory so dev and eval results cannot overwrite one another. Records include
source hashes/references, outputs, scores and grader evidence, model tiers and exact models used,
flags, retry policy, tool-contract hash, sandbox-image version, Python version, tokens and cost.

A finished primary record cannot be overwritten. Repeating a command refuses before model calls;
`ARGS="--resume"` keeps finished records and retries missing records and harness errors. One
process at a time can publish into a commit/config directory. Before replacing a harness error,
the exact prior record is archived under `attempts/{kind}/{system}/{case_id}/{run_id}.json`.
A crash after archival reuses that immutable archive on retry. A conflicting or unwritable archive
refuses replacement. Prior attempts stay outside primary/repeat rates, but their costs and timing
appear as recovery overhead and enter total spend and the aggregate spending cap.
Invalid/corrupt or misplaced records are refused. Cancellation preserves a harness-error record
with the known usage before re-raising the original interruption. An unfinished provider call
may have an unobserved charge: the remaining case budget is reserved conservatively in the
aggregate cap, separately from measured spend. Reports label partial usage and measured lower
bounds, rather than claiming unknown charges were zero.
`ARGS="--allow-dirty"` writes only to ignored `results/raw/`; these scratch records are never a
regression baseline. Commit code before collecting reportable evidence.

The optional repeat sample is separate:
`ARGS="--repeats 3 --repeat-cases case_a,case_b,case_c,case_d,case_e"`. At least five distinct
cases and three runs per case are required. They live in `repeats/{system}/{case_id}/{n}.json`
and enter none of the primary pass rates, paired comparisons, latency or cost summaries.

Setup, system work, grading, admission waits and retry sleeps are separate fields. The adapter
reports retry-sleep totals without changing frozen event payloads. If a system cannot report a
retry duration, it remains unknown; it is not represented as zero. List price comes from the
versioned price table. Actual spend is zero for Gemini dev free-tier calls, list price otherwise.

A system/grader failure creates a harness-error record and does not cancel the other cases.
Harness errors count as failures in primary summaries and are listed explicitly. Graded failed
cases remain immutable measurements. The summary includes Wilson 95% intervals, a paired outcome
table, a paired percentile bootstrap with 10,000 resamples and a fixed seed (inconclusive when the
interval contains zero), nearest-rank p95 and mean latency, and total/per-task list and actual cost.

## Regression comparison

`regressions.json`, `summary.json`, and `summary.md` are derived from per-case records. The runner
starts at the parent commit and finds the nearest first-parent ancestor with results under the
same config/profile directory. It prints the selected base and distance. If no base exists it
states that no regression check was possible. Cases with changed content hashes, split hashes,
run-config hashes, or unfinished measurements are listed as not comparable; they cannot silently support a quality comparison.
Newly failing cases are printed by system/case ID and saved in the report.

Exit status: 0 means every requested run was recorded and finished with no newly failing case;
1 means an unfinished/unrecorded run or a regression; 2 is preflight refusal; 130 is interruption.
A graded failure by itself is a valid measurement, and does not prevent collecting the rest.

## Current boundaries

The agent system uses the same controller as `hypora run` and reads final artifacts from its stored
records. It runs tabular cases and supports injected tool errors. Planted malformed-data input is
accepted by the case machinery, but causal grading is deferred to S3.H3 (D-032); these cases
cannot receive passing fault credit yet.
Unsupported fault types are refused rather than recorded as successfully injected. The full visible constraints are sent in the request and agent context. Secondary metrics and
quality targets flow through the existing TaskSpecification and ExperimentContract fields;
unsupported hard exclusions, named-model mandates, resource and optimization constraints refuse
before model calls. Later steps expand the product behavior;
this runner does not implement them. Synthetic data and fake systems appear only in unit tests.
Acceptance evidence uses the actual agent, baseline, graders, and real OpenML/UCI snapshots
(`tests/controller/test_eval_acceptance.py`; only the model is scripted there).

A run without a verified model reports the decision its approved plan states, or none if it stopped
before a plan; the grader recomputes that claim (D-032).

The optional trusted session permitted-tools field intersects the role allowlist. It cannot grant
any permission. With the allowlist defense off, the existing ablation still exposes all server
tools. Controlled pre-server tool faults perform only this role/case permission check, record that
specific policy decision, and return a transient error without executing the tool body. They do
not claim that inner argument, contract, budget, or firewall checks ran. Forbidden calls continue
to the server so its normal refusal cannot be masked by an injected error.

A minimal `pending/{run_id}.json` accounting receipt is written atomically before dispatch, with
only IDs, config/case hashes, and the reserved budget. If the process dies before a final record,
resume refuses with the run ID and budget exposure so unknown spend cannot disappear. An existing
matching final/partial record reconciles a receipt left by a crash after publication. This does
not resume or recover controller state.
