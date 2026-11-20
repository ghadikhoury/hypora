# S5.2 final run matrix: collector and run book

PLAN S5.2; spec §19.5; the frozen protocol in `docs/notes/test-set-protocol.md` (S5.1). Collector:
`scripts/final_eval.py`. No step here changes a prompt, flag, budget, model, grader, case or split.

## What is collected

| Arm | Config (results segment) | System | Paid | Primary | Repeat sample |
|---|---|---|---|---:|---:|
| `multi` | `default` | `agent` | yes | 26 | 6 x 3 |
| `single` | `single_agent` | `agent` | yes | 26 | 6 x 3 |
| `baseline` | `default` | `baseline` | no | 26 | 6 x 3 |

- 88 paid runs; 44 baseline runs (no model calls). The repeat sample is 3 *additional* runs of the
  protocol's six cases (`repeats/<system>/<case>/1..3.json`); the primary run is not repeat #1, and
  repeats never enter the primary rates or paired tables.
- Each arm runs once. The baseline is recorded once, in the `default` segment beside `multi`, and
  every report (multi vs baseline, single vs baseline) reads the same records.
- Records: `results/<freeze commit>/default/{agent,baseline}/...` and
  `results/<freeze commit>/single_agent/agent/...`; replaced harness errors under `attempts/`.
- Collection files: `results/final/s5.2-manifest.json` (commit, case hashes, config hashes, lock
  hash; a later pass with a different identity is refused), `s5.2-paid-accounting.json` (ledger
  audit, IDs and costs only), `s5.2-summary.{json,md}`, and `s5.2-orphans/` only if a collector was
  killed. Git-ignored state: `runs/final-s5.2/` (journal, lock, runs and state databases).

## Spending: one ledger, hard $10 cap

The existing D-054 ledger (`scripts/stage2_eval.py`), with its own journal and a $10 cap header.
Every actual OpenAI SDK attempt, retries included, reserves $0.36 (the Luna worst case) before
dispatch; a reservation is refused when measured spend plus outstanding reservations plus $0.36
would pass $10. Unknown outcomes keep their reservation forever; a definitive HTTP error with no
usage settles at $0.

The 88 case budgets add up to $52.80, more than the cap, so the runner's full-budget pre-check is
switched off for paid arms (`gate_plan`) and the per-attempt reservation is the hard limit, as for
S4.2f (D-052). A run the cap refuses (before it starts, by `CapGatedAgent`, or mid-run, by the
ledger) is a `harness_error`: not measured, excluded from Wilson intervals and paired tables,
listed by case, and retried only with `--resume` (the earlier attempt archived). Expected spend is
88 x $0.035 = about $3.

## Run book

Preconditions: S5.1 merged and tagged `eval-freeze`; Docker Desktop running; about 20 GB free disk.
Commands run in the WSL Ubuntu shell.

### 1. A clean checkout of the freeze (about 10 minutes)

```bash
git clone https://github.com/ghadikhoury/hypora.git ~/hypora-final && cd ~/hypora-final
git fetch --tags && git switch --detach eval-freeze
make setup
# Test-case data, identical bytes (the runner re-checks every SHA-256):
mkdir -p data && cp -a ~/hypora-s5.1/data/. data/
#   or fetch it fresh:  uv run --locked python -m hypora.evaluation.cases.test.generate
# Put OPENAI_API_KEY in ~/hypora-final/.env yourself (never paste it in chat).
```

If `scripts/final_eval.py` is part of the freeze commit, use `F="uv run --locked python -m
scripts.final_eval"`. Otherwise run the collector from outside the checkout, so the tree stays
clean and records name the freeze commit:

```bash
mkdir -p ~/s52-tools
git -C ~/hypora-s5.2 show origin/s5.2-core-eval:scripts/final_eval.py > ~/s52-tools/final_eval.py
F="env PYTHONPATH=$PWD uv run --locked python $HOME/s52-tools/final_eval.py"
```

### 2. Dry run (seconds; no Docker, key or provider)

```bash
$F plan
```

Check: `freeze identity OK`, `Readiness lock matches`, `26 cases`, `Case data 26/26`, both configs
on `openai/gpt-6-luna` with fallback False, `parallel limit here 2`, 88 paid runs, $10 headroom.

### 3. Launch order

Run each pass in the background with a log; `setsid nohup` survives closing the terminal.

```bash
mkdir -p runs/final-s5.2/logs
L=runs/final-s5.2/logs
# a. Baseline primary: free smoke test of Docker, data, splits and graders (about 0.2-1 h; unmeasured).
setsid nohup $F collect --arm baseline --pass primary --parallel 2 > $L/baseline-primary.log 2>&1 &
# b. After (a) ends: both paid arms side by side, one run each at a time (about 2-6 h).
setsid nohup $F collect --arm both --pass primary --parallel 2 > $L/paid-primary.log 2>&1 &
# c. After (b) ends with every primary graded: the paid repeat sample (about 1.5-4 h).
setsid nohup $F collect --arm both --pass repeat --parallel 2 > $L/paid-repeat.log 2>&1 &
# d. Baseline repeats (free, under 1 h); it shares the default segment, so never during b/c.
setsid nohup $F collect --arm baseline --pass repeat --parallel 2 > $L/baseline-repeat.log 2>&1 &
# e. Tables.
$F summarize
```

`--arm both` runs `multi` and `single` at the same time (one run each), so both systems see the
same hours and provider conditions and, if the cap were ever reached, neither loses its runs
alone. Per run: 5-13 minutes. Total paid wall time at 2 at once: about 3.7-9.5 hours.

### 4. Watching progress and spend

```bash
tail -f runs/final-s5.2/logs/paid-primary.log   # one line per finished run: [k/n] system/case
$F plan                                          # graded / retry / missing per arm and pass, ledger
$F spend                                         # republish results/final/s5.2-paid-accounting.json
```

`plan` reads the ledger without changing it: measured spend, unknown reserved, headroom. Measured
plus reserved can never pass $10.

### 5. Exit codes and resume

- `0`: every run of the pass graded and recorded. `1`: some runs are harness errors or unrecorded
  (listed at the end of the log). `2`: refused before anything ran (the reason is printed).
- **Interrupted (Ctrl-C, or a failed run):** rerun the same command with `--resume`. Graded records
  are kept and never re-run or re-spent; only harness errors and missing runs start again.
- **Killed abruptly (host low-memory reaper, WSL shutdown):** the runner refuses to resume while a
  run has an accounting receipt and no record. Its paid attempts are already in the ledger, so:
  ```bash
  docker ps --filter label=hypora.sandbox=1   # leftover sandbox jobs: docker stop <id>
  $F reconcile                        # lists orphaned runs and their ledger exposure
  $F reconcile --confirm              # keeps them in results/final/s5.2-orphans/, clears receipts
  $F collect --arm both --pass primary --parallel 2 --resume
  ```
  If the host kills a job for low memory, drop to `--parallel 1` (or `--arm multi`, then
  `--arm single`) before resuming.
- A cap refusal shows as harness errors with "cap reached" in the record's `error`. Raising the cap
  needs Ghadi's approval and a new journal; do not edit the ledger.

### 6. After collection

`$F summarize` writes `results/final/s5.2-summary.{json,md}` from the records (run it in the
collection checkout: the too-good-score section reads the runs' state databases). It reports per
system the pass rate over graded runs with a 95% Wilson interval and harness errors listed apart;
paired tables and a 10,000-resample paired bootstrap (seed 1990; "inconclusive" when the interval
includes 0) for multi vs baseline, single vs baseline and multi vs single; per-case outcomes and
hidden metrics; failure categories; mean and nearest-rank p95 latency; actual and list-price cost
per task with the price date; the repeat sample; and how often the too-good-score check fired on
non-leakage cases, the column it named and what it recommended (S5.1). Copy `results/<freeze>/` and
`results/final/` into the S5.2 branch for the PR.
