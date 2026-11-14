# Live-run hardening of the A/B/C workflow (D-045)

Until 2026-11-14 the A/B/C workflow had been exercised only with scripted models. The scripted
tests never produced a semicolon, a curly apostrophe, a missing citation or a cautious refusal, so
the first live runs on both profiles failed. This note lists each failure, its cause and its fix.
It is evidence for the failure analysis: these are real first-contact failures, not planted ones.

## Failures found live, in the order they appeared

| # | Where | Symptom | Cause | Fix |
|---|---|---|---|---|
| 1 | A's research decision | Every multi run FAILED after about 10 s (`CONTRACT: ValueError`) | The role-text check accepted only letters and `' ? . , ! -`, and banned the words `class`, `return`, `import`, `def`, `lambda`. Real assignments said "class imbalance", "return guidance", used `;`, `/`, `:` and `’` | Prose passes; code syntax, credentials and firewall-recognized data are refused; one bounded correction |
| 2 | C's worker proposal | `INVALID_MODEL_REQUEST` on Gemini | Gemini answers HTTP 400 to a response schema whose nested arrays carry `maxItems` | Strip `maxItems` from the schema sent to Gemini; the adapter still validates the full schema |
| 3 | First decision | `PROVIDER_SERVER_ERROR` ended the run | Three retries (about 12 s) did not outlast a brief provider overload | Five retries |
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
