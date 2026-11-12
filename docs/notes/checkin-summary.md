# Stage 2 check-in material

Snapshot for the Nov 4 check-in (S2.6); superseded by later stages.

> Stage 1/2 evidence and procedures. Preserve original model/config identities and results.
> Current implementation priorities and future evaluation runs are defined in PLAN (D-036).

Prepared for Ghadi to adapt and submit; the agent has not recorded or submitted the course check-in.

## What works

Hypora runs a single-agent tabular workflow on real public data. It profiles the dataset, builds
and approves a contract, executes bounded trials in isolated workers, scores the final model on
held-out data, and produces a report and trace. A separate fixed-pipeline baseline uses the same
evaluation and grading infrastructure.

The first recorded evaluation ran on 2026-11-08 UTC, at source commit
`ec81155d54c78568df93e7dd6751e9e90f1316e3`. Both the pinned eval profile and the development
profile produced the same six-case outcomes: the agent passed 3/6 and the baseline passed 2/6.
The paired difference was +1/6, with a 95% paired-bootstrap interval [0, 0.5], so this small
development comparison is inconclusive. Both matching S1.7 eval-profile smoke tasks produced
verified final models.

## What is next

Stage 3 adds data planning and methodology checks, then repeats the approved model-selection
procedure. Later stages integrate the remaining agent workflows and run the held-out evaluation.
The temporary `gpt-6-luna` eval lock is not a claim that it won the model-selection comparison.

## Two biggest risks

1. **Data methodology:** both systems fail the grouped-row, target-leakage and temporal-order
   cases. The current Stage 1 contract cannot express the needed data-planning choices; predictive
   metrics alone do not establish valid methodology.
2. **Evidence and generalization:** six planted dev cases are a small development sample, not the
   held-out test set. The interval includes zero; results do not establish a reliable advantage.

## Evidence

- [Pinned eval summary](../../results/ec81155d54c78568df93e7dd6751e9e90f1316e3/single_agent/summary.md)
- [Dev profile summary](../../results/ec81155d54c78568df93e7dd6751e9e90f1316e3/single_agent.dev/summary.md)
- [Matching S1.7 eval smoke tasks](../../results/stage1/eval/summary.md)
- [Paid attempt accounting](../../results/stage2/paid-accounting.json): $0.06326554 measured,
  $0 unknown reserved, against the approved $8 aggregate cap.

## Course handoff

- Record a 30-second to 2-minute video showing `hypora run` and `make eval`.
- Review and finalize the one-page update and its claims.
- Submit the update, video, and repository link at this step's merge commit.
