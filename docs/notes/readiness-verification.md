# S4.2j readiness verification (D-062)

S4.2j verifies the hardened A/B/C workflow on the six dev cases (as revised by D-055, D-059) with
the multi-agent `default` configuration on the pinned `eval` profile (`openai/gpt-6-luna` on every
tier, no fallback). The six dev cases are development controls, not held-out test results; they
were changed and fixed after seeing results (see Limitations). The frozen Stage 5 test set measures
generalization.

**Outcome.** The gate as the owner set it before verify4 (D-062: two runs with at least five of six
cases passing on one integrated commit) is met on `3cd984947dc3c02b62222fad53a931ec39908492`
(`main` after #82): verify4 passed 5/6 and verify5 passed 6/6. The plan's original "all six cases
pass in one run" is also met by verify5 alone, but verify4 on the same commit shows it does not
hold every time.

## Procedure

Every run used the same orchestrator script shape (`run-verify<N>.py`, outside the repository):
`build_plan` with `cases="dev"`, `config="default"`, `systems=("agent",)`, `profile=eval`,
`parallel=1`, agent only (no baseline arm); `run_eval` through the normal runner (same controller,
MCP server, worker, sandbox, hidden splits and programmatic graders as `make eval`), wrapped in the
S2.6 durable paid ledger (`scripts/stage2_eval.py` `SpendLedger`, $0.36 reserved before every
OpenAI SDK attempt, settled only from provider-reported usage). Each run had its own state
directory, journal and ledger cap, and asserted the expected clean commit before any call. The
earlier runs (diag-c5328d5, verify-2d7b1bf, verify1) kept the full-budget pre-check with ledger caps
of $3.85–$4.00. verify2 passed its $2.95 cap as the runner's `max_total_usd`, so the runner refused
it before any call (six $0.60 case budgets = $3.60). From verify2b on, the runner's limit was $3.60
and only the ledger cap, enforced by the per-attempt reservations, bounded spend: $2.95 (verify2b),
$2.92 (verify2c), then $2.75 under the owner's $3 per-run cap (verify3–5). Run state is outside the repository
(`~/hypora-readiness-verification/runs/s4.2j-<label>/`); none of it is committed except as listed
below.

## Committed evidence

| What | Path |
|---|---|
| verify4 summary, regression report | `results/3cd984947dc3c02b62222fad53a931ec39908492/s4.2j-verify4/default/` |
| verify5 summary, regression report | `results/3cd984947dc3c02b62222fad53a931ec39908492/s4.2j-verify5/default/` |
| verify4 paid audit (IDs, usage, prices) | `results/readiness/s4.2j-verify4-paid-accounting.json` |
| verify5 paid audit | `results/readiness/s4.2j-verify5-paid-accounting.json` |

The two runs share a commit and config, so each sits under its own `s4.2j-verify<N>` segment
instead of the runner's `<commit>/default/` path. That also keeps them out of the runner's
automatic regression base (`hypora/evaluation/runner/regression.py` looks for `<commit>/<config>/`).

**Per-case records are held back.** gitleaks' `generic-api-key` rule flags the `split_key` line in
10 of the 12 records, the D-035 false-positive class. All 12 split keys are identical to those in
the committed S4.2f records. The scanner exception in `.gitleaks.toml` is path-scoped, and each
earlier extension needed Ghadi's approval (D-052, D-053), so the records wait for that approval.
They are kept byte-identical in `~/hypora-s4.2j-records-3cd9849.tar.gz` (SHA-256
`04dce1030a5ee8d630e5a8c0dd70103a6d5fc0f49ea6002a9d7a62316176d7df`). Every number below was read
from those records and the ledgers.

## Identity (verify4 and verify5)

| Field | Value |
|---|---|
| Commit | `3cd984947dc3c02b62222fad53a931ec39908492`, clean (`dirty: false`) |
| Config / profile | `default` (`agent_mode: multi`), `eval` |
| Run config hash | `1fa3d2ab8aa04a854c60e5cd1b5bc0c4e177ca60d703918a734d5c8a102b8799` |
| Lock | `results/readiness/lock.json` at that commit, SHA-256 `94095df4154d8e0c6d8e3a9c74f1b3287347ce2c9b7542a4ad7bd206ecdac6e1`, `step: S4.2j` |
| Prompts | `agent_a_decide@v7`, `agent_a_intake@v2`, `agent_a_report@v2`, `agent_b@v3`, `agent_c@v13`, `agent_c_methodology@v8`, `agent_c_workers@v2`, `c_worker@v1`, `c_worker_final@v1`, `intake@v2`, `reporter@v1`, `role_plan@v1` (single-agent prompts unused); hashes in the lock |
| Tool contract | `1518dfe40ff96fbe0c7eb1dc94f8df685d2c706e17355910899b58d0ccea12e9` |
| Sandbox image / Python | `hypora-sandbox-tabular:f50772e217853fa1`, 3.12.14 |
| Model | `openai/gpt-6-luna`, every tier, temperature not configurable, no fallback calls |
| Retry | 3 retries, 6 s base, 60 s max, 240 s timeout; no retries occurred |
| Case hashes | imbalance `f8923a0d`, clean `873f6aaf`, grouped `4c7bf077`, leakage `d836aa0d`, temporal `f0e59e99`, tool failure `80249395` (D-059 limits) |

## verify4 and verify5, per case

| Case (bar) | verify4 run | Result | Hidden | $ | Calls | Tokens in/out | Wall s |
|---|---|---|---:|---:|---:|---|---:|
| imbalance (macro-F1 ≥ 0.71) | `run_050f17f2a10f` | pass | 0.791 | 0.0309 | 41 | 416,436 / 20,461 | 465 |
| clean control (MAE ≤ 0.76) | `run_b14f744a17f3` | pass | 0.609 | 0.0239 | 34 | 269,196 / 21,546 | 428 |
| grouped (MAE ≤ 8.0) | `run_545992c1d4e8` | pass | 7.708 | 0.0396 | 54 | 536,139 / 32,380 | 752 |
| leakage (ROC-AUC ≥ 0.65) | `run_19e5e0c7f4a9` | pass | 0.750 | 0.0461 | 54 | 521,041 / 39,369 | 672 |
| temporal (MAE ≤ 146) | `run_4b4ad8d832e1` | **fail** | — | 0.0342 | 39 | 327,847 / 31,991 | 361 |
| tool failure (ROC-AUC ≥ 0.67) | `run_41777ef6ad70` | pass | 0.757 | 0.0291 | 32 | 321,102 / 20,098 | 408 |
| **Total** | | **5/6** | | **0.2038** | **254** | 2,391,761 / 165,845 | 3,086 |

| Case | verify5 run | Result | Hidden | $ | Calls | Tokens in/out | Wall s |
|---|---|---|---:|---:|---:|---|---:|
| imbalance | `run_3dcbc13e413b` | pass | 0.794 | 0.0312 | 35 | 377,742 / 22,721 | 488 |
| clean control | `run_03ada46c0423` | pass | 0.607 | 0.0218 | 35 | 225,514 / 21,298 | 433 |
| grouped | `run_794c56b8f6eb` | pass | 7.708 | 0.0312 | 40 | 445,365 / 23,987 | 551 |
| leakage | `run_7b719d4c437b` | pass | 0.754 | 0.0404 | 40 | 376,607 / 33,891 | 581 |
| temporal | `run_092f9488371e` | pass | 72.83 | 0.0440 | 49 | 428,274 / 41,977 | 787 |
| tool failure | `run_38f48d453247` | pass | 0.751 | 0.0345 | 46 | 477,948 / 23,021 | 515 |
| **Total** | | **6/6** | | **0.2031** | **245** | 2,331,450 / 166,895 | 3,353 |

Pass rates: verify4 5/6, Wilson 95% [0.436, 0.970]; verify5 6/6, [0.610, 1.000]. Latency mean /
p95 (nearest-rank): verify4 514.4 s / 751.7 s, verify5 559.1 s / 786.6 s. No run reached a case
limit; no retries, rate-limit waits under 0.02 s, no unknown reservation, no harness error. Run
windows (UTC, 2026-11-21): verify4 12:52:58–13:40:55, verify5 13:42:43–14:34:49.

- **verify4 temporal.** C proposed the split; A's pre-lock concern named `casual` and `registered`,
  the platform locked the split and its joint target-composition test (D-060) flagged both as
  target components. C then nominated `trl_78e131c3`, and A chose to report before final
  verification. The platform facts recorded at that stop contradict a stop for an unrunnable
  candidate: its code hash matched the submitted job, the job completed both sandbox stages and the
  file parses. A's own reason is filtered from the record, so it cannot be quoted. No model was
  verified, so the graded failure is "run did not finish" and every model-dependent check.
- **Grouped (both runs)** passed through the D-060 constant-reference rule: no validated trial beat
  the training-fold median (mean validation MAE 12.09) by more than one paired standard error, so
  a constant trial was verified (verify4: a trial that already predicted the constant; verify5: a
  platform constant trial, chosen when A asked for the best validated trial). Both scored the same
  hidden MAE, 7.708. This is the outcome D-055's bar was set to allow; it is not a model that
  learned the task.

## Every S4.2j paid run

Revised cases (D-055 and later), oldest first. "Killed" means the process was stopped by the Claude
Code low-memory process reaper: the log ends after its first line, no traceback, and a pending
record remains.

| Label | Commit | Window (UTC) | Cases done | Result | Ledger cap | Spend $ |
|---|---|---|---|---|---:|---:|
| diag-c5328d5 | `c5328d5` | 11-20 06:12–06:42 | 2 of 6, killed in grouped (`run_220184d2ab2c`) | imbalance pass (0.827), clean pass (0.614) | 4.00 | 0.1118 |
| verify-2d7b1bf | `2d7b1bf` | 11-20 14:51–14:54 | 0, stopped in imbalance (`run_7f87921b4241`) | none | 3.85 | 0.0214 + 2 unsettled |
| verify1 | `2d7b1bf` | 11-20 14:51–15:38 | 6 | **4/6** | 3.88 | 0.1903 |
| verify2 | `4663caf` | 11-20 | 0 | refused before any call by the full-budget pre-check ($3.60 > $2.95) | 2.95 | 0 |
| verify2b | `4663caf` | 11-20 20:40–20:43 | 0, killed in imbalance (`run_808f9e1bf591`) | none | 2.95 | 0.0221 |
| verify2c | `d9cef81` | 11-20 21:09–21:57 | 4, killed in temporal (`run_cb30d113647c`) | 3 pass, grouped fail | 2.92 | 0.1769 |
| verify3 | `a5d32af` | 11-21 01:14–02:10 | 6 | **4/6** | 2.75 | 0.2161 |
| verify4 | `3cd9849` | 11-21 12:53–13:41 | 6 | **5/6** | 2.75 | 0.2038 |
| verify5 | `3cd9849` | 11-21 13:43–14:35 | 6 | **6/6** | 2.75 | 0.2031 |

Total measured spend on the revised cases: **$1.1456** (plus two unsettled verify-2d7b1bf
reservations, at most $0.72 by the reservation bound). For killed runs without an audit file the
spend is the sum of settled journal entries.

Failures in the incomplete and earlier runs:
- **verify-2d7b1bf** started 8 s before verify1 on the same commit, so for about three minutes two
  paid runs overlapped. It was stopped during its first case with 24 attempts reserved and 22
  settled. Its cause is not recorded in the log.
- **verify1** (`run_f8f3054c80e0` imbalance): macro-F1 0.516 < 0.71 (intake had dropped the user's
  sentence about available features, D-058) and a `no_policy_violations` failure ("job … has no
  trustworthy execution boundary"); whether the overlap with verify-2d7b1bf contributed is not
  established. Temporal (`run_8e0edd3ee50c`): no final model, A's pre-lock concern deadlocked
  (D-058). Passes: clean 0.630, grouped 6.814, leakage 0.732, tool failure 0.754.
- **verify2c** grouped (`run_3aae18d950dd`): `max_compute_seconds` 615 of 600, no final model
  (D-059). Passes: imbalance 0.826, clean 0.617, leakage 0.746.
- **verify3** grouped (`run_1f0ab125d08c`): hidden MAE 8.837 > 8.0, no constant trial (D-060).
  Temporal (`run_8545505bdf03`): MAE 67.07 met the bar but the final model kept `casual`, a target
  component (D-060). Passes: imbalance 0.820, clean 0.608, leakage 0.754, tool failure 0.770.

Before the D-055 revision (old case files, old limits, not counted toward the gate), S4.2j also made
paid runs on `85e8cd1` (11-19, 4 cases before an interruption; $0.1048 measured, $0.36 unknown),
`2c2332f` (three attempts during a provider outage: $0.0011 measured with $3.96 held per journal,
the D-054 ledger bug; a third, `-run3`, $0.2019 settled, 7 unsettled, killed in tool failure; and
one tool-failure-only rerun, pass, $0.0293) and `db9034a` (11-20, 2/6, $0.2086). Their findings
drove D-054 to D-056. Measured total: $0.5458.

## Limitations

- **The dev cases are not an unbiased measurement.** Cases, bars and limits (D-055, D-059) and the
  workflow (D-056, D-058, D-060, and D-061 if PR #83 merges) were changed after seeing these
  results. The frozen Stage 5 test set measures generalization.
- **Run-to-run nondeterminism.** Same commit, config and cases: verify4 5/6, verify5 6/6. Temporal
  failed in verify1, verify3 and verify4 for three different reasons. Two runs cannot estimate a
  pass rate; the per-run Wilson intervals overlap widely.
- **Grouped passes only through the platform's constant rule** (above).
- **The #72 `max_trials <= 4` guidance** (`agent_c`, `agent_c_workers`: three focused trials, no
  workers) is not exercised: every dev case now has `max_trials` 6 (D-055).
- **Reaper-killed runs** (diag-c5328d5, verify2b, verify2c; verify-2d7b1bf ended with the same
  signature, cause not recorded) left pending records and are reported, not resumed or graded.
- **Per-case records are not yet in the repository** (gitleaks approval, above).
- Agent only, no baseline arm and no single-agent arm; one model; one WSL2 machine, one run at a
  time. `results/readiness/lock.json` still says `verification: pending` because
  `tests/unit/test_readiness_eval.py` asserts that value; changing it is a code change outside this
  docs PR.
