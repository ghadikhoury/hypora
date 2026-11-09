# Stage 2 collection (S2.6)

> Stage 1/2 evidence and procedures. Preserve original model/config identities and results.
> Current implementation priorities and future evaluation runs are defined in PLAN (D-036).

The temporary `eval` model is `openai/gpt-6-luna` on every tier, temperature not configurable,
no fallback (D-033). Dev remains Gemini Flash-Lite on its free tier with no fallback. This is an
owner-approved temporary lock, not a claim that Luna won D-007's eligibility-gated comparison.
The Stage 1 deterministic contract cannot pass leakage/group methodology traps; all failures
remain visible, and model selection is repeated after Stage 3 data planning exists.

Commit source/model changes first, then run from a clean WSL checkout with the existing local
credentials, verified snapshots and generated cases. The collector uses the same real controller,
MCP, worker, sandbox, hidden splits and programmatic graders as `make eval`. It has no scripted
provider or test-double mode. The runner command still supports normal comparisons; this S2.6
entry point additionally shares the paid cap across comparisons and matching smoke tasks.

```bash
uv run --locked python -m hypora.evaluation.cases.dev.generate
flock /tmp/hypora-engineering-heavy.lock uv run --locked python scripts/stage2_eval.py run --profile dev
flock /tmp/hypora-engineering-heavy.lock uv run --locked python scripts/stage2_eval.py run --profile eval
flock /tmp/hypora-engineering-heavy.lock uv run --locked python scripts/stage2_eval.py smoke
```

Each comparison runs all six cases on agent and baseline once. A finished graded result is not
silently repeated; use `--resume` only for unfinished measurements. The matching smokes actually
use the pinned eval profile on the same two S1.7 tasks/snapshot hashes. Earlier dev and one-off
Luna smoke records remain unchanged. A matching finished smoke is retained on restart; an
unfinished intent blocks restart until its run ledger/evidence is reconciled. No recorded result
is relabeled. The handout's repeated held-out-test sample is collected at the later S5.2 gate;
these six dev cases do not substitute for held-out tests.

## Paid exposure

One append-only journal under ignored `runs/stage2/paid-attempts.jsonl` admits every actual OpenAI
SDK attempt, including adapter retries and both smokes. File locking serializes admission across
processes, and file plus directory-chain fsync precedes dispatch. The SDK performs no hidden
retries. Paid requests explicitly use Standard service and the global OpenAI endpoint, preventing
project auto-tier or endpoint overrides from invalidating the price bound. The cache lifetime is
the provider default of 30 minutes. There is no model-parameter/output-limit change to either profile.

Before each attempt, reserve $0.36. Luna's verified absolute maximum is below that: 1.05M input
tokens at the highest long-context input/cache-write price ($0.25/M), plus 128K output tokens at
$0.75/M, is $0.3585. Admission requires known charges plus all outstanding reservations plus the
next reservation to remain at or below the owner-approved $8 aggregate cap. The six $0.60 case
budgets and two $0.45 smoke budgets remain in place; per-attempt admission also covers overshoot,
retries, ambiguous failures and future invocations. It cannot raise the cap.

Only validated provider-reported usage settles a reservation; unknown usage, cancellations,
timeouts and crashes keep their full exposure. Malformed/truncated history refuses paid calls.
Published accounting cannot be overwritten to hide a missing historical journal. An uncertain
case becomes an unfinished measurement with reserved exposure, rather than a passing budget
claim. The sanitized audit under `results/stage2/paid-accounting.json` contains per-attempt counts
(including cache writes), exact prices/source/date, measured charges and unknown reserves. It
contains no prompts, raw data, credentials or machine paths.

The public case records retain all outcomes and uncertainty intervals. A bootstrap interval
including zero is inconclusive. The first run may have no comparable parent results; the report
states that limitation. Targeted local checks, required CI and independent review remain required before merge (D-034). No check-in materials are submitted by the agent.
