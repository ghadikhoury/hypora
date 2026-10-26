# Cost and token estimate (S1.7)

Spec: §16 (cost and latency), §17 (providers), §19.5 (cost per task). Plan: S1.7, §1 model policy.
Decisions: D-007 (dev model; eval model chosen at S2.6), D-016 (S1.7 on dev), D-027 (one-off
gpt-6-luna smoke comparison). Evidence: `results/stage1/` (per-run records, reports, summary).

Every run below is **dev / unrecorded**: a check that the system works end to end on real data,
not an evaluation and not a model selection. **Measured** numbers come from the four committed
runs. **Projected** numbers price those measured token counts at another model's list price; they
are planning figures, not measurements. Nothing here chooses the eval model: S2.6 does that by cost
per passed trap (D-007).

## What was run

`scripts/stage1_smoke.py`, on 2026-10-26, from commit `f09645f`, through the `hypora run` path
(preflight, controller, MCP server, worker, sandbox, verifier, reporter) on the `dev` profile, with
plan approval by policy (D-019). Same two tasks for both models; the snapshot references and
SHA-256s are fixed in the script, so S2.6 can add matching `eval` evidence (D-016).

| Task | Snapshot | Rows × columns | Target | Task type, metric |
|---|---|---|---|---|
| credit-g | OpenML 46918 v5, CC BY 4.0 | 1000 × 21 | `good_or_bad_customer` | binary classification, ROC-AUC |
| fish-toxicity | OpenML 46954 v8 (QSAR fish toxicity), CC BY 4.0 | 907 × 7 | `LC50` | regression, MAE |

Costs are list prices from `configs/prices.yaml` (Flash-Lite and gpt-6-luna, read 2026-10-24).
Gemini ran on the free tier, so its **actual spend was $0**; OpenAI bills its list price, so the
Luna figures are real spend. **Total actual spend for S1.7: $0.0324** (both Luna runs).

## Measured: gemini-3.5-flash-lite (dev profile as configured)

| | credit-g | fish-toxicity | Mean per run |
|---|---|---|---|
| Outcome | DONE, verified, ROC-AUC 0.7348 | DONE, verified, MAE 0.6097 | |
| Limits fired | none | none | |
| Model calls (agent turns) | 26 (24) | 54 (52) | 40 |
| Input tokens (cached share) | 496,640 (77.7%) | 988,799 (79.0%) | 742,720 |
| Output tokens | 4,871 | 5,570 | 5,220 |
| Cost at list price | $0.0570 | $0.0996 | $0.0783 |
| Largest single-call input | 31,899 | 30,052 | |
| Model time per call | 30.2 s | 17.4 s | |
| Wall clock | 815 s | 1,245 s | 1,030 s |
| Tool calls (of which status polls) | 22 (7) | 50 (28) | |
| Trials; jobs | 4; 5 completed, 1 failed | 7; 8 completed | |
| Events (bytes) | 172 (106,695) | 325 (203,946) | 248 (155,320) |

The failed job on credit-g was a trial whose `fit` stage exited with an error (`STAGE_FAILED`, not
retryable); the agent went on with its other trials and the run still ended verified.

**Run-to-run variation.** A pilot pass of the same two tasks, made earlier the same day through
the same code path but before the script was committed (records kept locally, not in
`results/`), used 12 and 38 model calls, 117,283 and 645,671 input tokens, and $0.0237 and
$0.0757. The same task took about half the calls in one run that it took in the other: one run
per task is a rough average, not a stable cost. In the pilot's credit-g run, one Gemini call
failed with a server error after 3 retries; the single agent stopped, and the controller
finished the run with the best validated trial (verified), as designed.

## Measured: gpt-6-luna (one-off smoke comparison, D-027)

Every `dev` tier mapped to `openai/gpt-6-luna` in memory; the `eval` profile was not used.

| | credit-g | fish-toxicity | Mean per run |
|---|---|---|---|
| Outcome | DONE, verified, ROC-AUC 0.7671 | DONE, verified, MAE 0.6458 | |
| Limits fired | **`max_llm_tokens`** | **`max_llm_tokens`** | |
| Model calls (agent turns) | 54 (53) | 56 (55) | 55 |
| Input tokens (cached share) | 1,004,017 (97.0%) | 1,010,643 (97.6%) | 1,007,330 |
| Output tokens | 5,940 | 6,095 | 6,018 |
| Cost (billed) | $0.0164 | $0.0160 | $0.0162 |
| Largest single-call input | 28,996 | 27,797 | |
| Model time per call | 2.4 s | 2.4 s | |
| Wall clock | 224 s | 234 s | 229 s |
| Tool calls (of which status polls) | 84 (74) | 110 (100) | |
| Trials; jobs | 3; 3 completed, 1 cancelled | 3; 3 completed, 1 cancelled | |
| Events (bytes) | 435 (255,289) | 541 (316,339) | 488 (285,814) |

Both runs were stopped by the run's token limit (1,000,000 LLM tokens, `configs/budgets.yaml`);
the controller cancelled the unfinished job, chose the best validated trial, scored it once on the
test split, and the verifier passed it. The Reporter's narrative was skipped because the limit had
already fired, so these two reports have no narrative (D-019 4a behavior). Because the limit cut
the runs short, **their token counts and cost are capped by the limit, not what a full run costs**.

The single estimate given before these runs ($0.1538 for both, from the Gemini token counts at
Luna's price with no caching) overstated the cost about 5x: OpenAI cached 97% of the input.

## Projections for the other S2.6 candidates (not measured)

Method: price each measured Gemini call's tokens at the candidate's list price, then average per
run. "Gemini cache share" bills the cached tokens Gemini reported at the cached price; "no
caching" bills every input token at the full price (an upper bound). Gemini Flash prices are from
https://ai.google.dev/gemini-api/docs/pricing (page updated 2026-10-22, read 2026-10-26): Standard
paid tier, the same for gemini-3.6-flash and gemini-3.8-flash, **$0.75 input / $0.075 cached /
$3.75 output per 1M tokens through 2026-12-31, doubling to $1.50 / $0.15 / $7.50 from
2027-01-01**. Assumptions: the candidate needs as many calls and tokens as Flash-Lite did, and
writes as much output (a model that thinks more writes more output tokens).

| Model | Basis | Per run | ~500 runs |
|---|---|---|---|
| gemini-3.5-flash-lite (paid tier) | measured tokens, measured list cost | $0.0783 | $39.14 |
| gpt-6-luna | measured Luna runs (cut short by the token limit) | $0.0162 | $8.10 |
| gpt-6-luna | projected: Gemini tokens, Gemini cache share | $0.0244 | $12.18 |
| gpt-6-luna | projected: Gemini tokens, no caching | $0.0769 | $38.44 |
| gemini-3.6/3.8-flash, 2026 price | projected: Gemini tokens, Gemini cache share | $0.1826 | $91.31 |
| gemini-3.6/3.8-flash, 2026 price | projected: Gemini tokens, no caching | $0.5766 | $288.31 |
| gemini-3.6/3.8-flash, 2027 price | projected: Gemini tokens, Gemini cache share | $0.3653 | $182.63 |

The recorded runs (S2.6, Stages 4–5) and the demo all fall before 2027-01-01, so the 2026 Flash
price applies to them.

## What ~500 recorded runs would take

- **Shape of these runs.** Stage 1 runs are single-agent, tabular, about 1,000 rows. The full
  graph (S4.1) adds Intake, Research and Data leads with parallel workers, trial workers, a
  Reviewer, and the Supervisor, so a recorded multi-agent run will make more model calls than
  these. **Treat the per-run figures above as a floor** and re-measure at the Stage 2 gate (S2.6),
  where the dev case set runs on both profiles.
- **Planning range from these measurements** (500 runs × per-run cost): about $8–38 on
  gpt-6-luna, about $39 on Flash-Lite paid, about $91 on Gemini Flash (2026 price, with caching),
  before any multi-agent increase. If a full-graph run costs as much as 3 Stage-1 runs, which is a
  planning assumption and not a measurement, these become about $24–115, $117, and $274.
- **Free-tier throughput.** At the adapter's self-imposed 450 requests/day on Flash-Lite
  (D-007), a 40-call run allows about 11 runs a day: 500 runs would take about 44 days. The free
  tier can carry development and dev-profile gates, not hundreds of recorded runs before Dec 1.
- **Wall clock.** About 17 minutes per run on free-tier Flash-Lite (17–30 s per call), about 4
  minutes on Luna (2.4 s per call); 500 sequential runs take about 143 h and 32 h. Runs share no
  state (spec §9.1), so the runner (S2.4) can run several at once within the provider limits.
- **Event log.** About 600 bytes per event and 250–490 events per run: 500 runs add about
  78–143 MB to the state database.

## Behavior differences between Gemini (dev) and gpt-6-luna

An early warning for the model policy (PLAN.md §1): the same code, prompts, and tasks behave
differently enough on another model to matter for the recorded evaluation.

1. **Luna busy-polls job status.** The single-agent prompt (`single_agent@v1`, step 4) says to call
   `get_experiment_status` "until `finished` is true". Flash-Lite answers in 17–30 s, so its polls
   are naturally spaced (7 and 28 polls); Luna answers in 2.4 s and polled 74 and 100 times. Each
   poll is a full model turn that re-sends the conversation (about 17–30K tokens), so Luna
   consumed the run's 1M-token budget mostly waiting, and the limit fired before it could try more
   than 3 trials or reach the Reporter's narrative.
2. **Prompt caching.** Luna reported 97% of input tokens as cached, Gemini 78–79%; at Luna's cached
   price, 1M tokens cost $0.016. A token limit that counts cached and uncached tokens alike stops
   Luna runs at a very low cost.
3. **Breadth.** Gemini ran 4 and 7 trials; Luna 3 each, one of which was still unfinished when
   the limit fired and its job was cancelled.
4. **Outcomes.** Both models produced verified models on both tasks; test scores differ (credit-g
   ROC-AUC 0.7671 Luna vs 0.7348 Gemini; fish-toxicity MAE 0.6458 Luna vs 0.6097 Gemini, lower is
   better). With one run per task and different trials, these differences are not evidence that
   either model is better.
5. **Reliability.** No Luna call failed or was retried. Gemini had one retried call in the
   committed runs and one server error that outlasted its retries in the pilot (handled as above).

## Follow-ups (recommendations, not made here)

- **Status polling.** Two fixes are possible: let `get_experiment_status` wait for a change (an
  MCP tool signature change: stop-and-ask trigger 1, since the tool contract is frozen), or revise
  the prompt to space polls or batch them. Recommended before the multi-agent runs, whose trial
  workers get the same status tool (S3.C2); the planning stream (S3.H) or the prompt review
  (S4.2) are the natural places, subject to Ghadi's decision.
- **The 1M-token run limit.** The run guard counts input plus output tokens, cached included.
  Both Luna runs hit the limit, and Gemini fish-toxicity used 994,369 of the 1,000,000 (99.4%).
  The default budgets are provisional until S4.2 (`🧑 HUMAN · DECIDE`), which should set the
  token limit with these numbers in hand and decide whether cached tokens count toward it.
- **Re-measure at S2.6.** The Stage 2 gate runs the dev case set on both profiles; its records
  replace these per-run figures for budgeting.

## Reproduce

```bash
make snapshot SOURCE=openml ID=46918 && make snapshot SOURCE=openml ID=46954
uv run --locked python scripts/stage1_smoke.py run gemini                 # free tier
uv run --locked python scripts/stage1_smoke.py estimate                   # Luna estimate
uv run --locked python scripts/stage1_smoke.py run luna --confirm-paid    # PAID, $1 cap
uv run --locked python scripts/stage1_smoke.py summary
```

State goes to `runs/stage1/` (git-ignored); `hypora trace <run_id> --db runs/stage1/hypora.sqlite`
shows any run's event log. The Luna variant refuses to start without `--confirm-paid` and keeps
the total OpenAI spend recorded in that database under the cap (D-027).
