# Configuration and feature flags (S0.4)

**Layout.** `configs/` (spec §22.1) holds the configuration *data*; `hypora/core/config/` is the
*loader* that validates it. The spec layout has no loader package; it lives under `core/` like the
other runtime code (same reasoning as D-001).

```
configs/*.yaml + HYPORA_* env vars --load_settings()--> HyporaSettings (validated, frozen)
                                     --RunConfig.build()--> RunConfig (deeply frozen, hashed)
                                     --snapshot_json()--> canonical JSON for the runs row
```

**Precedence** (lowest → highest): code defaults < `configs/*.yaml` < `HYPORA_*` environment
variables < explicit overrides (how an eval config in `configs/eval/<name>.yaml` applies its flags).
Sources are merged field by field: overriding `flags.defenses.firewall` keeps every other flag.

| File | Top-level key | Contents |
|---|---|---|
| `flags.yaml` | `flags` | spec §19.4 switches + §9.5 parallelism limits |
| `models.yaml` | `models` | `dev` / `eval` profiles: provider + model + temperature per tier, fallback policy; per-model `rate_limits` and the `retry` policy (S0.5) |
| `resources.yaml` | `resources` | pool overrides (`null` = derive from hardware), memory reserve, LLM concurrency |
| `budgets.yaml` | `budgets` | default run `Budget` and per-job `JobLimits` (S0.2 schema types) |
| `eval/<name>.yaml` | — | `name`, `model_profile`, partial `flags` |

Env names: `HYPORA_` + path joined by `__`, e.g. `HYPORA_FLAGS__AGENT_MODE=single`,
`HYPORA_MODEL_PROFILE=eval`, `HYPORA_CONFIG_DIR=/path/to/configs`.
Config-directory variable spellings follow the same case-insensitive lookup as the settings
source; an explicit directory argument still wins.

**Fail fast.** Configuration errors raise `ConfigError` with setting and source context. Value
validation errors name the setting, bad value, file, and environment variable. That covers invalid
enum values, unknown keys (in YAML or env), a file that
doesn't hold exactly its own section, malformed YAML, and missing files. Malformed complex
environment JSON and malformed named-eval files also return `ConfigError` with the environment
settings or offending file named. Pydantic-settings silently ignores env vars that match no setting,
so `load_settings` rejects any `HYPORA_*` variable whose
first segment isn't a section; a typo like `HYPORA_FLAG__AGENT_MODE` can't go unnoticed.

**YAML 1.1 booleans.** PyYAML reads a bare `on`/`off`/`yes` as booleans. On/off flags therefore
accept booleans (`planning: on` works), and no field is named `on` (the fallback triggers field is
`triggers`).

**Recorded eval runs ignore nothing silently.** `load_settings_for_eval` refuses to run if any
`HYPORA_*` variable other than `HYPORA_CONFIG_DIR` is set, so a leftover `export` can't turn an
"all defenses on" comparison into a partial one. No env var may change the pinned eval model
profile (`HYPORA_MODELS__EVAL__*`) even in dev, because it is also dev's fallback target.
This includes any case variant of an `eval` key inside a bulk JSON `HYPORA_MODELS` value
(`eval`, `EVAL`, `Eval`, etc.). Both variable names and JSON profile keys are checked using the
settings source's case-insensitive semantics. A bulk override containing only `dev` (in any case)
still works and preserves the pinned eval mapping. Regression tests cover direct loads, a real
active eval `RunConfig` with all three non-placeholder IDs, and named eval loads against the
installed settings source's normalization.
`RunConfig.build` refuses an active profile that still has `placeholder-*` model IDs.

**Enum reuse.** `approval_policy` is `schemas.ApprovalPolicy`; profiles, tiers, and providers are
`schemas.ModelProfile`, `ModelTier`, `Provider`; budgets are `schemas.Budget` / `JobLimits`. Only
switch enums with no schema counterpart (`AgentMode`, `RetrievalMode`, …) live in `flags.py`.

**Hardware detection** never raises and never imports torch: affinity-aware CPU count, RAM from
`sysconf`, GPUs from `nvidia-smi --query-gpu=...` with a timeout. No `nvidia-smi`, a timeout, a
non-zero exit, or unparseable output all mean "0 GPUs" plus a recorded status. Pools follow spec
§9.2 (`cpu = cores − 1`, `gpu = detected`, `memory = RAM − reserve`); overrides may shrink a pool
but never exceed the hardware. When RAM is known, an explicit memory override must fit within
**RAM minus the reserve**, and a reserve that leaves no usable RAM is rejected even with an
override. If RAM cannot be detected, an explicit memory pool is required and is taken as the
operator's capacity declaration. Pool values retain their precision: display rounding must never
increase scheduler capacity or round a small positive override to zero.

**RunConfig** is a `FrozenRecord`: every nested model is frozen and every sequence a tuple, so
nothing can change after `build()`. It records the effective flags, both model profiles, budgets,
resolved pools, and detected hardware; `hash` is its canonical-JSON SHA-256.

**Provisional values.** S0.5 set the dev model IDs (D-007); the eval IDs stay `placeholder-*`
until S2.6 `🧑 DECIDE`. `configs/prices.yaml` is not a settings section: the LLM adapter loads it
(`hypora.core.llm.pricing`, see llm-adapter.md).
Budgets and worker limits are placeholders for S4.2 `🧑 DECIDE`.
