# Live-run hardening of the A/B/C workflow (D-045)

Until 2026-11-14 the A/B/C workflow had been exercised only with scripted models. The scripted
tests never produced a semicolon, a curly apostrophe, a missing citation or a cautious refusal, so
the first live runs on both profiles failed. This note lists each failure, its cause and its fix.
It is evidence for the failure analysis: these are real first-contact failures, not planted ones.

## Failures found live, in the order they appeared

| # | Where | Symptom | Cause | Fix |
|---|---|---|---|---|
| 1 | A's research decision | Every multi run FAILED after about 10 s (`CONTRACT: ValueError`) | The role-text check accepted only letters and `' ? . , ! -`, and banned the words `class`, `return`, `import`, `def`, `lambda`. Real assignments said "class imbalance", "return guidance", used `;`, `/`, `:` and `’` | Prose passes; code syntax, credentials, instruction-like text, record IDs and firewall-recognized data are refused; one bounded correction |
| 2 | C's worker proposal | `INVALID_MODEL_REQUEST` on Gemini | Gemini answers HTTP 400 to a response schema whose nested arrays carry `maxItems` | Strip `maxItems` from the schema sent to Gemini; the adapter still validates the full schema |
| 3 | First decision | `PROVIDER_SERVER_ERROR` ended the run | Three retries with a 2 s base (about 12 s) did not outlast a brief provider overload | A 6 s base backoff (about 20–40 s over the same three retries) |
| 4 | B's strategy report | "Research returned invalid or unsupported citations" in most runs | B listed a path (often the hosted API) that no returned source covered, or copied a citation detail inexactly, and the report was discarded without feedback | One correction with the rule; an uncovered path may be listed with every rating unknown |
| 5 | C's data contract (Luna) | Schema refusal on `split.validation` | The model wrote a sentence where the field is a code (`holdout`, `stratified_kfold_5`) | The prompt lists the codes (`agent_c_methodology@v2`) |
| 6 | C's data contract (Luna) | `MethodologyError: unresolved methodology questions` | C asked which label is positive, a fact it is never shown | One correction with the rule; the prompt context states what the platform guarantees |
| 7 | A's methodology review (Luna) | A chose `report` | A treated withheld target statistics as an ambiguity | A's review context states the platform's guarantees and what counts as a conflict |
| 8 | C in Phase B (Luna) | C stopped without a trial | A's assignment asked for folds and caps the contract fixes differently; C followed A literally | A is told the contract fixes split, validation, metric and limits; C's prompt (`agent_c@v4`) says the contract wins and to submit at least a baseline |
| 9 | A after failed research (Luna) | A chose `report` | Unavailable research looked like a reason to stop | The unavailable observation says research is advisory |
| 10 | Reporter | Narrative refused twice | Placeholders such as `{{final.test_metrics.mae}}`; the refusal only said "no digits" | The refusal names the two valid placeholder forms |

The report's leakage line also said "No target was provided" on every run although the user named
one: the profile is built without the target on purpose. It now says why the profile does not
screen and where leakage is examined.

## What did not change

The boundary is still enforced by code. A correction request contains only the fixed text of the
rule that refused the output, never the output itself, and each author gets one correction before
the run stops safely as before. The data firewall, the allowlists, the approval gate, the verifier
and the hidden test split are unchanged.

## Live evidence (credit-g unless noted; runs are local and git-ignored)

- Gemini (`dev`): `run_7a28af0a0b62`, `run_4787880871f8` (credit-g), `run_a340ef0d15d3` (QSAR fish
  toxicity, regression), `run_07fa7fdc10f5` (planted target leakage), `run_c8e6591e07f9`
  (Parkinson's grouped rows): all DONE with a verified model.
- Luna (`eval`): `run_9a6029d4c1be` DONE with a verified model after failures 5–9 were fixed; it
  exercised B's correction, C's correction and a second research round.
- The planted-leakage run kept the planted post-outcome column (test ROC-AUC 0.97 against about
  0.76 without it). The run completed correctly as software; catching the column is a methodology
  quality question for the evaluation (`dev_target_leakage`), not a crash.

## Robustness sweep (D-047)

Mining all 23 saved runs after PR #45 (traces, model calls, reports) found one run-ending bug and
a set of platform-caused failures that live runs survived only by luck.

| # | Where | Symptom | Cause | Fix |
|---|---|---|---|---|
| 11 | A's pre-final code review (Luna) | `run_b2d5f594bf4b` ended with no verified model: A said the code "cannot be parsed" | The excerpt cut every line at 200 characters without a mark; a 215-character line of valid, already-executed code lost its closing brackets. Every saved Luna candidate had such a line; Gemini's code never exceeded about 111 characters | Lines are shown whole or replaced by a counted marker; limits sized for long PyTorch scripts; full lines are screened before anything is left out; A gets code-checked `platform_facts` (hash, job, parse) and `agent_a_decide@v4` says a marker is not a defect |
| 12 | C's locked-split diagnostics (Luna) | 20 of 74 tool results failed (`INVALID_ARGUMENTS at run_diagnostic`) in three runs; no diagnostic ever ran | C copied `dataset:ds_main` from the task specification into `dataset_id` in twenty parallel calls; the pattern error did not say what to send | The error's `suggested_recovery` says to omit `dataset_id`; `agent_c_methodology@v3` says so too; identical failures in one turn share one replan instead of two plan calls |
| 13 | B's research | `MODEL_EVIDENCE_UNAVAILABLE` (18 times), retried and listed as unresolved | Gated Hugging Face cards answer HTTP 401; the platform called that a transient outage | `MODEL_EVIDENCE_NOT_READABLE` (missing data, not retried, choose another reference) |
| 14 | report.md | Twenty identical "Fault ... unresolved" lines | One line per fault; "unresolved" unexplained | Grouped with a count; each outcome explained (report.json unchanged) |
| 15 | Validation | Most runs compared models on one holdout fold of 1,000 rows (validation 0.87, test 0.756) | The methodology prompt listed holdout first and gave no size guidance | Below 10,000 rows C is advised to prefer 5-fold cross-validation |
| 16 | Silent cuts elsewhere | Trial facts cut mid-JSON at 20,000 characters; tool results cut mid-JSON at 60,000; log lines, error messages, model cards, passages and B's summary cut without a count | Plain slicing | `hypora/core/bounded_text.py`: whole lines, whole items, counted markers, valid JSON |
| 17 | Longer jobs | A 5-fold final job could be cancelled as timed out; jobs still running after C stopped were cancelled after 15 minutes | The final job's timeout assumed 10 stages; the drain had a fixed bound | Stages counted from the contract's folds; the drain waits up to the jobs' own limits, bounded by the run's wall clock |
| 18 | Error reports | "CONTRACT: LLMError", "Research model failed: LLMCallError" | Non-platform exceptions were reduced to their type | Platform-written detail: the refusal rule or the LLM error code and retries |
| 19 | B's correction turn (Luna, found by this PR's first live run `run_6e893a2f8062`) | `INVALID_MODEL_REQUEST` (HTTP 400 in 0.2 s) on both research rounds; research unavailable | Before replaying B's refused answer, the platform redacted credential shapes in the whole message, including OpenAI's encrypted reasoning. That ciphertext is URL-safe base64, where `sk-…` and `hf_…` shapes occur by chance in a long blob; the rewritten blob was rejected | `LLMAdapter.redact_message`: readable text is still redacted; opaque provider tokens (`encrypted_content`, Gemini thought signatures) are replayed byte-for-byte. Same fix in the adapter's schema-repair retry |
| 20 | A's pre-final review (Luna, same run) | A chose `report`: "B's revised research must be reconciled before acting" | The notes said "A asked B to revise its research" and the revision had failed; nothing said A had already reconciled it | A note records the completed reconciliation; the pre-final review says notes are history and research is advisory |

Earlier failures in the saved runs (role-text refusals, Gemini `maxItems`, schema refusals, the
`ExceptionGroup` from worker proposals) predate PR #45 and are fixed there.

### Live evidence for the sweep (credit-g demo)

- Luna `run_6e893a2f8062` (first commit): no verified model; surfaced failures 19 and 20. A
  judged the shown code sound, so the code-review failure (11) did not recur.
- Luna `run_6cd1967c79cf` (fixes 11–20): DONE with a verified model; `stratified_kfold_5`
  (validation ROC-AUC 0.803, test 0.758; the holdout runs had 0.869 and 0.756); no argument
  failures; gated cards not retried; final job 12 sandbox stages.
- Gemini `run_9117c90a5261` (`dev`, same code): DONE with a verified model; `stratified_kfold_5`
  (validation 0.803, test 0.756).

### Found, not fixed here

- Waiting on a job costs one model turn per `get_experiment_status` call (at most 45 s each, and
  the 45 s is in the frozen tool description). A 30-minute training job would need about 40 turns
  and resend the whole context each time, against a 60-turn cap and a 1M-token budget. Long
  training needs a platform-side wait that does not spend model turns: a design change for a
  later step. (Addressed by D-049: the wait grows to 10 minutes; a 30-minute job costs 6 calls.)
- The provider request timeout is 120 s; the slowest saved call took 73 s. Larger reviews may get
  closer; the retry policy lives beside the `eval` profile, so it was left for Ghadi. (Addressed
  by D-049: 240 s.)
- Sandbox log tails stay bounded by the launcher's own tail limit; a long training log keeps only
  its end.
