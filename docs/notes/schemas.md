# Core schemas: conventions (S0.2)

Package: `hypora/core/schemas/`, one module per concept group, all re-exported from `__init__.py`.

| Module | Contents |
|---|---|
| `common.py` | `StrictModel`, `Record`, ID/ref types, shared enums, `Budget`, `JobLimits`, `CostEstimate`, `ResourceConstraints` |
| `task.py` | `TaskSpecification`, `DatasetProfile`, `RouteDecision` (stages 1–3) |
| `strategy.py` | `StrategyReport` (§7.1) |
| `contracts.py` | `DataContract` (§7.2), `ApprovedPlan`, `ExperimentContract`, `ApprovalRecord` |
| `briefs.py` | `Brief`, `TrialBrief`, `OptimizationBrief`, `Handoff` |
| `findings.py` | four worker findings, `VerificationResult`, `ReviewVerdict` |
| `decisions.py` | `SupervisorDecision`, `TrialBatchProposal`, `LeadDecision` |
| `trials.py` | `TrialSpec`, `ProposedTrial`, `TrialFile` (`trial.yaml`), trial-directory constants |
| `tools.py` | `ToolName`, `ToolTier`, `ToolError`, `SubmitExperimentRequest` |
| `plan.py` | `Plan` (F4) |
| `events.py` | `EventType` and one event class per type; `AnyEvent` discriminated union |

**Rules every schema follows**

- `extra="forbid"`, `allow_inf_nan=False`: unknown fields and NaN fail loudly.
- Every top-level record (`Record`) carries `schema_version` (`Literal[1]`); any other value fails.
- Enums are `StrEnum` with the spec's spelling (`DATA_PROBLEM`, `group_temporal`, `ResearchFinding`).
- Cross-field rules live in validators that reject exactly the case they exist for, with one test each
  (`tests/unit/schemas/test_validators.py`). Examples: group/temporal splits need their keys and
  matching folds; a text column never lists values (data firewall); the `eval` profile never records
  a fallback; a verification result's `passed` must agree with its checks.
- Values that can be computed are computed, not stored: metric direction (`MetricName.direction`),
  batch and plan totals (`total_estimated_cost`), hashes (`content_hash()`).

**LLM outputs vs. internal records.** Schemas an LLM fills (`SupervisorDecision`, `LeadDecision`,
`ReviewVerdict`, findings, reports) are flat: an action enum plus the fields that action needs,
checked by validators. This keeps their JSON Schemas simple for provider structured-output modes.
Internal records use discriminated unions where they help type-checking, most importantly events:
`parse_event(row)` returns `ModelCallEvent`, `PlanEvent`, …, so `event.payload` is typed.

**Findings never carry metric numbers.** Workers reference results by ID (`result:trl_0002/validation`).
Every number in a report comes from platform-computed results (spec principle 3, §5.2 Report).

**Choices worth knowing**

- `ToolError` adds a required `error_class` to the §12 example (the spec lists the classes; S3.H2 maps
  them to retry/replan/ask). Spec §12 updated.
- Event `tool` fields are validated identifier strings, not `ToolName`: the log must be able to record
  an attempted call to a tool that does not exist.
- `PinnedModelRef` (`name@revision`) is required wherever weights are run or downloaded; a user-named
  model without a revision is a `ModelName`.
- `ExperimentContract` and `ApprovedPlan` are frozen; changes are new versions with new hashes.

**Known follow-up for S3.H1.** Its Done-when says "the first event after role start is always a plan".
The plan is produced by a model call, and `model_call` is logged when that call returns, so the literal
sequence is `role_started → model_call → plan`. S3.H1 should confirm the intended reading (the plan
precedes the role's first `tool_call`) before implementing.

Regenerate `docs/schemas/` with `make schemas`; a test fails if it is stale.
