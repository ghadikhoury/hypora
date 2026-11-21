# Held-out test cases v1 (S5.1)

26 tabular cases on pinned real UCI snapshots, none shared with the dev set. They are the frozen
Stage 5 measurement; the protocol, thresholds, exclusions and freeze identity are in
[the test-set protocol](../../../../docs/notes/test-set-protocol.md). YAML requests and constraints
expose neither trap metadata nor hidden rows. Every case has the dev set's limits (D-055, with
compute raised to 1,800 s as in #81): $0.60, 1,800 compute seconds, 2M tokens, six trials, 200 tool
steps and 1,800 wall seconds, with matching contract bounds; hidden split 25%, seed 20261120.

| Case | Real source | Family | Construction |
|---|---|---|---|
| `test_leak_heart_failure` | UCI 519 Heart Failure Clinical Records | Leakage | `time` (follow-up length) is known only after the outcome. |
| `test_leak_student_dropout` | UCI 697 Predict Students' Dropout and Academic Success | Leakage | Prediction at the end of semester 1; the six 2nd-semester columns come later. |
| `test_leak_ai4i_failure` | UCI 601 AI4I 2020 Predictive Maintenance | Leakage + imbalance | The five failure-mode flags describe the failure; 3.4% positive. |
| `test_leak_student_grades` | UCI 320 Student Performance | Leakage | Prediction before the year; period grades `G1`, `G2` come later. |
| `test_leak_bank_marketing` | UCI 222 Bank Marketing | Leakage + imbalance | Prediction before the call; `duration` is the call's length; 11.7% positive. |
| `test_group_concrete` | UCI 165 Concrete Compressive Strength | Groups | plantings.py: `mix_id` from the seven component amounts (427 mixtures). |
| `test_group_airfoil` | UCI 291 Airfoil Self-Noise | Groups | plantings.py: `setup_id` from angle, chord, velocity, thickness (106 setups). |
| `test_group_real_estate` | UCI 477 Real Estate Valuation | Groups | plantings.py: `location_id` from latitude and longitude (259 locations). |
| `test_time_metro_traffic` | UCI 492 Metro Interstate Traffic Volume | Temporal | Original hourly rows; hidden rows are the latest timestamps. |
| `test_time_beijing_pm25` | UCI 381 Beijing PM2.5 | Temporal | plantings.py: ISO `timestamp` from the clock columns; hours with no reading dropped. |
| `test_time_tetouan_power` | UCI 849 Power Consumption of Tetouan City | Temporal | plantings.py: ISO `DateTime`; zones 2 and 3 (other targets) dropped. |
| `test_imb_htru2` | UCI 372 HTRU2 | Imbalance | 9.2% pulsars; no primary metric given. |
| `test_imb_bankruptcy` | UCI 572 Taiwanese Bankruptcy Prediction | Imbalance | 3.2% bankrupt; no primary metric given. |
| `test_imb_churn` | UCI 563 Iranian Churn | Imbalance | 15.7% churn; no primary metric given. |
| `test_imb_credit_default` | UCI 350 Default of Credit Card Clients | Imbalance | 22.1% default; no primary metric given. |
| `test_clean_abalone` | UCI 1 Abalone | Clean control | Original rows. |
| `test_clean_power_plant` | UCI 294 Combined Cycle Power Plant | Clean control | Original rows. |
| `test_clean_dry_bean` | UCI 602 Dry Bean | Clean control | Original rows, seven classes. |
| `test_clean_rice` | UCI 545 Rice (Cammeo and Osmancik) | Clean control | Original rows. |
| `test_clean_breast_cancer` | UCI 17 Breast Cancer Wisconsin (Diagnostic) | Clean control | Original rows. |
| `test_fault_submit_spambase` | UCI 94 Spambase | Tool fault | First `submit_experiment` returns `TOOL_UNAVAILABLE`; recovery required. |
| `test_fault_submit_superconductivity` | UCI 464 Superconductivity Data | Tool fault | First `submit_experiment` returns `RATE_LIMITED`; recovery required. |
| `test_fault_write_magic` | UCI 159 MAGIC Gamma Telescope | Tool fault | First `write_trial_files` returns `TOOL_UNAVAILABLE`; recovery required. |
| `test_fault_submit_banknote` | UCI 267 Banknote Authentication | Tool fault | First `submit_experiment` returns `WORKER_UNAVAILABLE`; recovery required. |
| `test_fault_malformed_maternal` | UCI 863 Maternal Health Risk | Malformed data | plantings.py blanks 24 seeded labels; a causal safe stop is required. |
| `test_fault_malformed_landsat` | UCI 146 Statlog (Landsat Satellite) | Malformed data | plantings.py writes class codes as text and blanks 60 seeded labels; safe stop. |

All sources declare CC BY 4.0 on their UCI page. Snapshot and planted bytes stay under ignored
`data/`; `data/snapshots/manifest.json` records source URL, version, license, retrieval date and
SHA-256 on each machine, and the protocol note copies those fields. A changed download fails the
pinned case hash and needs investigation, never substitution.

Fetch the snapshots, then regenerate and validate every case:

~~~bash
for id in 1 17 94 146 159 165 222 267 291 294 320 350 372 381 464 477 492 519 545 563 \
          572 601 602 697 849 863; do make snapshot SOURCE=uci ID=$id; done
uv run --locked python -m hypora.evaluation.cases.test.generate
~~~

Re-derive the bars from the visible rows (`thresholds.py`; no hidden row is read, no model provider
is called):

~~~bash
uv run --locked python -m hypora.evaluation.cases.test.thresholds
~~~

`derivation.json` records that output; `tests/unit/evaluation/test_test_cases.py` checks that every
case file's threshold and tolerance are the rule applied to it.
