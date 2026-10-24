# Configuration and feature flags (S0.4)

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
| `models.yaml` | `models` | `dev` / `eval` profiles: provider + model + temperature per tier, fallback policy |
| `resources.yaml` | `resources` | pool overrides (`null` = derive from hardware), memory reserve, LLM concurrency |
| `budgets.yaml` | `budgets` | default run `Budget` and per-job `JobLimits` (S0.2 schema types) |
| `eval/<name>.yaml` | — | `name`, `model_profile`, partial `flags` |

Env names: `HYPORA_` + path joined by `__`, e.g. `HYPORA_FLAGS__AGENT_MODE=single`,
`HYPORA_MODEL_PROFILE=eval`, `HYPORA_CONFIG_DIR=/path/to/configs`.

**Fail fast.** Every problem raises `ConfigError` naming the setting, the bad value, the file, and the
env var that sets it. That covers: invalid enum values, unknown keys (in YAML or env), a file that
doesn't hold exactly its own section, malformed YAML, and missing files. pydantic-settings silently
ignores env vars that match no setting, so `load_settings` rejects any `HYPORA_*` variable whose
first segment isn't a section; a typo like `HYPORA_FLAG__AGENT_MODE` can't go unnoticed.

**YAML 1.1 booleans.** PyYAML reads a bare `on`/`off`/`yes` as booleans. On/off flags therefore
accept booleans (`planning: on` works), and no field is named `on` (the fallback triggers field is
`triggers`).

**Enum reuse.** `approval_policy` is `schemas.ApprovalPolicy`; profiles, tiers, and providers are
`schemas.ModelProfile`, `ModelTier`, `Provider`; budgets are `schemas.Budget` / `JobLimits`. Only
switch enums with no schema counterpart (`AgentMode`, `RetrievalMode`, …) live in `flags.py`.

**Hardware detection** never raises and never imports torch: affinity-aware CPU count, RAM from
`sysconf`, GPUs from `nvidia-smi --query-gpu=...` with a timeout. No `nvidia-smi`, a timeout, a
non-zero exit, or unparseable output all mean "0 GPUs" plus a recorded status. Pools follow spec
§9.2 (`cpu = cores − 1`, `gpu = detected`, `memory = RAM − reserve`); overrides may shrink a pool
but never exceed the hardware.

**RunConfig** is a `FrozenRecord`: every nested model is frozen and every sequence a tuple, so
nothing can change after `build()`. It records the effective flags, both model profiles, budgets,
resolved pools, and detected hardware; `hash` is its canonical-JSON SHA-256.

**Provisional values.** Model IDs are `placeholder-*` (S0.5 fills dev; S2.6 `🧑 DECIDE` pins eval).
Budgets and worker limits are placeholders for S4.2 `🧑 DECIDE`.
