"""The summary of one eval (spec §19.5; PLAN.md S2.4): `summary.json` and `summary.md`.

Everything is computed from the records on disk, so a summary can be regenerated at any time and
says nothing the records don't. A `harness_error` counts as a failed case in every rate: it is a
missing pass, and leaving it out would flatter a system that crashes on hard cases. The summary
lists those cases so the numbers can't be read as finished.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from pydantic import Field

from hypora.core.schemas.common import StrictModel
from hypora.evaluation.runner.records import HARNESS_ERROR_CATEGORY, CaseRecord
from hypora.evaluation.runner.regression import RegressionReport
from hypora.evaluation.stats import (
    BootstrapVerdict,
    cost_per_task,
    latency_summary,
    paired_bootstrap_difference,
    paired_outcome_table,
    wilson_interval,
)


class Rate(StrictModel):
    """A pass rate with its 95% Wilson interval."""

    passed: int
    tasks: int
    rate: float
    wilson_low: float
    wilson_high: float


class Latency(StrictModel):
    """Seconds over a system's runs; the method names how p95 was taken."""

    mean: float
    p95: float
    method: str


class SystemSummary(StrictModel):
    """One system's results over the primary runs."""

    system: str
    pass_rate: Rate
    harness_errors: list[str] = Field(default_factory=list)
    latency_system: Latency
    latency_active: Latency
    list_cost_total_usd: float
    list_cost_per_task_usd: float
    actual_spend_total_usd: float
    actual_spend_per_task_usd: float
    model_calls: int
    input_tokens: int
    output_tokens: int
    rate_limit_wait_seconds: float
    retry_wait_seconds: float | None
    setup_seconds: float


class PairedComparison(StrictModel):
    """Two systems on the cases both ran (spec §19.5)."""

    first: str
    second: str
    cases: int
    both_pass: int
    first_only: int
    second_only: int
    both_fail: int
    difference: float
    ci_low: float
    ci_high: float
    resamples: int
    seed: int
    verdict: BootstrapVerdict
    unpaired_cases: list[str] = Field(default_factory=list)


class CaseRow(StrictModel):
    """One case's outcome per system."""

    case_id: str
    outcomes: dict[str, str]


class RepeatRow(StrictModel):
    """One case of the repeat sample, reported apart from the primary runs (spec §19.5)."""

    system: str
    case_id: str
    runs: int
    passed: int
    mean_latency_seconds: float
    mean_cost_usd: float


class ModelPriceLine(StrictModel):
    """A model that was called, with where its list price comes from."""

    provider: str
    model_id: str
    price_retrieved_on: str | None


class RecoveryOverhead(StrictModel):
    """Prior runs retained after resume, excluded from quality rates but never from spend."""

    system: str
    attempts: int
    list_cost_usd: float
    actual_spend_usd: float
    unobserved_spend_reserved_usd: float
    usage_complete: bool
    system_seconds: float
    setup_seconds: float
    grading_seconds: float
    rate_limit_wait_seconds: float
    retry_wait_seconds: float | None


class EvalSummary(StrictModel):
    """The whole of `summary.json`."""

    commit: str
    dirty: bool
    config: str
    profile: str
    flags: dict[str, object]
    run_config_hashes: list[str]
    sandbox_images: list[str]
    tools_contract_sha256: list[str]
    python_versions: list[str]
    tiers: dict[str, object]
    retry_policy: dict[str, object]
    systems: list[SystemSummary]
    paired: PairedComparison | None = None
    failure_categories: dict[str, dict[str, int]]
    cases: list[CaseRow]
    repeats: list[RepeatRow] = Field(default_factory=list)
    models: list[ModelPriceLine] = Field(default_factory=list)
    regression: RegressionReport
    recovery_overhead: list[RecoveryOverhead] = Field(default_factory=list)
    total_list_cost_usd: float = 0.0
    total_actual_spend_usd: float = 0.0
    total_unobserved_spend_reserved_usd: float = 0.0
    usage_complete: bool = True


def _rate(passes: Sequence[bool]) -> Rate:
    interval = wilson_interval(sum(passes), len(passes))
    return Rate(
        passed=sum(passes),
        tasks=len(passes),
        rate=sum(passes) / len(passes),
        wilson_low=interval.low,
        wilson_high=interval.high,
    )


def _latency(seconds: Sequence[float]) -> Latency:
    return Latency(**{k: v for k, v in asdict(latency_summary(seconds)).items() if k != "count"})


def _system_summary(system: str, records: Sequence[CaseRecord]) -> SystemSummary:
    list_cost = cost_per_task([r.usage.cost_usd for r in records])
    actual = cost_per_task([r.usage.actual_spend_usd for r in records])
    retry_waits = [r.timings.retry_wait_seconds for r in records]
    known_retry_waits = [wait for wait in retry_waits if wait is not None]
    return SystemSummary(
        system=system,
        pass_rate=_rate([r.passed for r in records]),
        harness_errors=[r.case_id for r in records if not r.finished],
        latency_system=_latency([r.timings.system_seconds for r in records]),
        latency_active=_latency([r.timings.active_seconds for r in records]),
        list_cost_total_usd=list_cost.total_usd,
        list_cost_per_task_usd=list_cost.per_task_usd,
        actual_spend_total_usd=actual.total_usd,
        actual_spend_per_task_usd=actual.per_task_usd,
        model_calls=sum(r.usage.model_calls for r in records),
        input_tokens=sum(r.usage.input_tokens for r in records),
        output_tokens=sum(r.usage.output_tokens for r in records),
        rate_limit_wait_seconds=sum(r.timings.rate_limit_wait_seconds for r in records),
        retry_wait_seconds=sum(known_retry_waits)
        if len(known_retry_waits) == len(records)
        else None,
        setup_seconds=sum(r.timings.setup_seconds for r in records),
    )


def _paired(
    first: str, second: str, by_system: Mapping[str, Mapping[str, CaseRecord]]
) -> PairedComparison | None:
    ones, twos = by_system[first], by_system[second]
    shared = sorted(set(ones) & set(twos))
    if not shared:
        return None
    a = [ones[case].passed for case in shared]
    b = [twos[case].passed for case in shared]
    table = paired_outcome_table(a, b)
    bootstrap = paired_bootstrap_difference(a, b)
    return PairedComparison(
        first=first,
        second=second,
        cases=table.total,
        both_pass=table.both_pass,
        first_only=table.first_only,
        second_only=table.second_only,
        both_fail=table.both_fail,
        difference=bootstrap.difference,
        ci_low=bootstrap.interval.low,
        ci_high=bootstrap.interval.high,
        resamples=bootstrap.resamples,
        seed=bootstrap.seed,
        verdict=bootstrap.verdict,
        unpaired_cases=sorted(set(ones) ^ set(twos)),
    )


def _pair_order(systems: Sequence[str]) -> tuple[str, str] | None:
    """The agent against the baseline when both ran; otherwise the first two systems by name."""
    if "agent" in systems and "baseline" in systems:
        return "agent", "baseline"
    return (systems[0], systems[1]) if len(systems) >= 2 else None


def _outcome(record: CaseRecord) -> str:
    if not record.finished:
        return "harness error"
    return "pass" if record.passed else "fail"


def build_summary(
    primary: Mapping[tuple[str, str], CaseRecord],
    repeats: Sequence[CaseRecord],
    regression: RegressionReport,
    attempts: Sequence[CaseRecord] = (),
) -> EvalSummary:
    """Compute the summary. Refuses an empty set: there is nothing to state a rate about."""
    if not primary:
        raise ValueError("a summary needs at least one primary record")
    current_ids = {record.run_id for record in (*primary.values(), *repeats)}
    # A crash after archival but before replacement leaves the same run in both places. Count
    # it once, and call it recovery overhead only after a new run actually replaced it.
    prior = {record.run_id: record for record in attempts if record.run_id not in current_ids}
    all_runs = {record.run_id: record for record in (*primary.values(), *repeats, *prior.values())}
    prior_systems: dict[str, list[CaseRecord]] = {}
    for record in prior.values():
        prior_systems.setdefault(record.system, []).append(record)
    by_system: dict[str, dict[str, CaseRecord]] = {}
    for (system, case_id), record in sorted(primary.items()):
        by_system.setdefault(system, {})[case_id] = record
    names = sorted(by_system)
    order = _pair_order(names)
    first = next(iter(primary.values()))
    provenance = [record.provenance for record in primary.values()]
    categories: dict[str, Counter[str]] = {}
    for (system, _), record in primary.items():
        for category in record.failure_categories:
            categories.setdefault(category, Counter())[system] += 1
    case_ids = sorted({case_id for _, case_id in primary})
    models = {
        (m.provider.value, m.model_id, m.price_retrieved_on)
        for record in all_runs.values()
        for m in record.usage.by_model
    }
    grouped: dict[tuple[str, str], list[CaseRecord]] = {}
    for record in repeats:
        grouped.setdefault((record.system, record.case_id), []).append(record)
    return EvalSummary(
        commit=first.provenance.commit_sha,
        dirty=first.provenance.dirty,
        config=first.provenance.config,
        profile=first.provenance.profile.value,
        flags=first.provenance.flags.model_dump(mode="json"),
        run_config_hashes=sorted({p.run_config_hash for p in provenance}),
        sandbox_images=sorted({p.sandbox_image for p in provenance}),
        tools_contract_sha256=sorted({p.tools_contract_sha256 for p in provenance}),
        python_versions=sorted({p.python_version for p in provenance}),
        tiers=first.provenance.tiers.model_dump(mode="json"),
        retry_policy=first.provenance.retry_policy.model_dump(mode="json"),
        systems=[_system_summary(name, list(by_system[name].values())) for name in names],
        paired=_paired(*order, by_system) if order else None,
        failure_categories={
            name: dict(sorted(c.items())) for name, c in sorted(categories.items())
        },
        cases=[
            CaseRow(
                case_id=case_id,
                outcomes={
                    name: _outcome(by_system[name][case_id])
                    for name in names
                    if case_id in by_system[name]
                },
            )
            for case_id in case_ids
        ],
        repeats=[
            RepeatRow(
                system=system,
                case_id=case_id,
                runs=len(runs),
                passed=sum(r.passed for r in runs),
                mean_latency_seconds=latency_summary([r.timings.system_seconds for r in runs]).mean,
                mean_cost_usd=cost_per_task([r.usage.cost_usd for r in runs]).per_task_usd,
            )
            for (system, case_id), runs in sorted(grouped.items())
        ],
        models=[
            ModelPriceLine(provider=provider, model_id=model_id, price_retrieved_on=retrieved)
            for provider, model_id, retrieved in sorted(models, key=str)
        ],
        regression=regression,
        total_list_cost_usd=sum(record.usage.cost_usd for record in all_runs.values()),
        total_actual_spend_usd=sum(record.usage.actual_spend_usd for record in all_runs.values()),
        total_unobserved_spend_reserved_usd=sum(
            record.usage.unobserved_spend_reserved_usd for record in all_runs.values()
        ),
        usage_complete=all(record.usage.complete for record in all_runs.values()),
        recovery_overhead=[
            RecoveryOverhead(
                system=name,
                attempts=len(runs),
                list_cost_usd=sum(record.usage.cost_usd for record in runs),
                actual_spend_usd=sum(record.usage.actual_spend_usd for record in runs),
                unobserved_spend_reserved_usd=sum(
                    record.usage.unobserved_spend_reserved_usd for record in runs
                ),
                usage_complete=all(record.usage.complete for record in runs),
                system_seconds=sum(record.timings.system_seconds for record in runs),
                setup_seconds=sum(record.timings.setup_seconds for record in runs),
                grading_seconds=sum(record.timings.grading_seconds for record in runs),
                rate_limit_wait_seconds=sum(
                    record.timings.rate_limit_wait_seconds for record in runs
                ),
                retry_wait_seconds=(
                    sum(record.timings.retry_wait_seconds or 0 for record in runs)
                    if all(record.timings.retry_wait_seconds is not None for record in runs)
                    else None
                ),
            )
            for name, runs in sorted(prior_systems.items())
        ],
    )


def _verdict_text(paired: PairedComparison) -> str:
    if paired.verdict is BootstrapVerdict.INCONCLUSIVE:
        return "inconclusive: the interval includes 0"
    higher = paired.first if paired.verdict is BootstrapVerdict.FIRST_HIGHER else paired.second
    return f"{higher} has the higher pass rate"


def _seconds(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1f}"


def _headline(summary: EvalSummary) -> list[str]:
    lines = [
        f"# Evaluation summary: {summary.config}",
        "",
        f"- Commit: `{summary.commit}`"
        + (" (dirty tree: not a recorded result)" if summary.dirty else ""),
        f"- Config: `{summary.config}`, profile `{summary.profile}`",
        f"- Run config hash: {', '.join(f'`{h[:16]}`' for h in summary.run_config_hashes)}",
        f"- Sandbox image: {', '.join(f'`{i}`' for i in summary.sandbox_images)}",
        "- Tool contract SHA-256: "
        f"{', '.join(f'`{h[:16]}`' for h in summary.tools_contract_sha256)}",
        f"- Python: {', '.join(summary.python_versions)}",
        f"- Tier models: `{summary.tiers}`",
        f"- Retry policy: `{summary.retry_policy}`",
        f"- Flags: `{summary.flags}`",
        "",
    ]
    errors = [(s.system, case) for s in summary.systems for case in s.harness_errors]
    if errors:
        listed = ", ".join(f"{system}/{case}" for system, case in errors)
        lines += [
            f"**{len(errors)} run(s) did not finish: {listed}. They count as failures below. "
            "Run again with `--resume` to replace them.**",
            "",
        ]
    return lines


def _rates(summary: EvalSummary) -> list[str]:
    lines = [
        "## Pass rates",
        "",
        "| System | Passed | Rate | 95% Wilson interval |",
        "|---|---|---|---|",
    ]
    for s in summary.systems:
        r = s.pass_rate
        lines.append(
            f"| {s.system} | {r.passed}/{r.tasks} | {r.rate:.3f} | "
            f"[{r.wilson_low:.3f}, {r.wilson_high:.3f}] |"
        )
    return [*lines, ""]


def _paired_lines(summary: EvalSummary) -> list[str]:
    p = summary.paired
    if p is None:
        return ["## Paired comparison", "", "Not available: it needs two systems on one case.", ""]
    lines = [
        "## Paired comparison",
        "",
        f"| Both pass | {p.first} only | {p.second} only | Both fail |",
        "|---|---|---|---|",
        f"| {p.both_pass} | {p.first_only} | {p.second_only} | {p.both_fail} |",
        "",
        f"Pass-rate difference ({p.first} minus {p.second}) over {p.cases} case(s): "
        f"{p.difference:+.3f}, 95% paired-bootstrap CI [{p.ci_low:+.3f}, {p.ci_high:+.3f}] "
        f"({p.resamples} resamples, seed {p.seed}, percentile method). **{_verdict_text(p)}.**",
        "",
    ]
    if p.unpaired_cases:
        lines += [f"Not paired, ran on one side only: {', '.join(p.unpaired_cases)}.", ""]
    return lines


def _cost_latency(summary: EvalSummary) -> list[str]:
    method = summary.systems[0].latency_system.method
    lines = [
        "## Latency and cost",
        "",
        f"Seconds per run; p95 by the {method} method. Active time is the run less rate-limit and "
        "retry waits. Setup and grading are not latency. List price is from configs/prices.yaml; "
        "actual spend is what was billed ($0 on the Gemini free tier).",
        "",
        "| System | Mean | p95 | Active mean | Active p95 | Rate-limit wait | Retry wait | Setup "
        "| List $/task | List $ total | Actual $/task | Actual $ total "
        "| Model calls | Tokens in/out |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary.systems:
        lines.append(
            f"| {s.system} | {s.latency_system.mean:.1f} | {s.latency_system.p95:.1f} | "
            f"{s.latency_active.mean:.1f} | {s.latency_active.p95:.1f} | "
            f"{s.rate_limit_wait_seconds:.1f} | {_seconds(s.retry_wait_seconds)} | "
            f"{s.setup_seconds:.1f} | ${s.list_cost_per_task_usd:.4f} | "
            f"${s.list_cost_total_usd:.4f} | ${s.actual_spend_per_task_usd:.4f} | "
            f"${s.actual_spend_total_usd:.4f} | {s.model_calls} | "
            f"{s.input_tokens:,}/{s.output_tokens:,} |"
        )
    lines.append("")
    if summary.models:
        lines += ["Prices:", ""]
        lines += [
            f"- {m.provider}/{m.model_id}: list price read {m.price_retrieved_on or 'unknown'}"
            for m in summary.models
        ]
        lines.append("")
    return lines


def _recovery_lines(summary: EvalSummary) -> list[str]:
    lines = [
        "## Total spend and recovery overhead",
        "",
        f"All primary, repeat, and archived attempts: list ${summary.total_list_cost_usd:.4f}; "
        f"actual ${summary.total_actual_spend_usd:.4f}. Each run ID is counted once.",
        "",
    ]
    if not summary.usage_complete:
        lines += [
            "**Usage is incomplete for interrupted/unreadable runs. Measured costs above are "
            "lower bounds; unknown charges are not reported as $0.**",
            f"Extra amount reserved for possible unobserved charges: "
            f"${summary.total_unobserved_spend_reserved_usd:.4f} (not measured spend).",
            "",
        ]
    if not summary.recovery_overhead:
        return [*lines, "No prior attempts were replaced.", ""]
    lines += [
        "Archived attempts contribute no extra primary observations or paired outcomes.",
        "",
        "| System | Prior attempts | List $ | Actual $ | Unknown reserved $ "
        "| Run s | Setup s | Grading s "
        "| Rate wait s | Retry wait s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary.recovery_overhead:
        lines.append(
            f"| {row.system} | {row.attempts} | ${row.list_cost_usd:.4f} | "
            f"${row.actual_spend_usd:.4f} | ${row.unobserved_spend_reserved_usd:.4f} | "
            f"{row.system_seconds:.1f} | {row.setup_seconds:.1f} | "
            f"{row.grading_seconds:.1f} | {row.rate_limit_wait_seconds:.1f} | "
            f"{_seconds(row.retry_wait_seconds)} |"
        )
    return [*lines, ""]


def _failures(summary: EvalSummary) -> list[str]:
    names = [s.system for s in summary.systems]
    lines = ["## Failure categories", ""]
    if not summary.failure_categories:
        return [*lines, "None.", ""]
    lines += [f"| Category | {' | '.join(names)} |", "|---|" + "---|" * len(names)]
    for category, counts in summary.failure_categories.items():
        cells = " | ".join(str(counts.get(name, 0)) for name in names)
        lines.append(f"| {category} | {cells} |")
    return [*lines, ""]


def _cases(summary: EvalSummary) -> list[str]:
    names = [s.system for s in summary.systems]
    lines = ["## Cases", "", f"| Case | {' | '.join(names)} |", "|---|" + "---|" * len(names)]
    for row in summary.cases:
        cells = " | ".join(row.outcomes.get(name, "not run") for name in names)
        lines.append(f"| {row.case_id} | {cells} |")
    return [*lines, ""]


def _repeat_lines(summary: EvalSummary) -> list[str]:
    if not summary.repeats:
        return []
    lines = [
        "## Repeat sample",
        "",
        "Reported apart from the primary runs (spec §19.5); it enters none of the numbers above.",
        "",
        "| System | Case | Passed | Mean seconds | Mean list $ |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.system} | {r.case_id} | {r.passed}/{r.runs} | {r.mean_latency_seconds:.1f} | "
        f"${r.mean_cost_usd:.4f} |"
        for r in summary.repeats
    ]
    return [*lines, ""]


def _regression_lines(summary: EvalSummary) -> list[str]:
    report = summary.regression
    lines = ["## Regressions", "", report.message, ""]
    for title, refs in (
        ("Newly failing", report.newly_failing),
        ("Recovered", report.recovered),
        ("Not comparable", report.not_comparable),
        ("New cases", report.new_cases),
        ("Missing from this run", report.missing_cases),
    ):
        if refs:
            lines += [f"**{title}:** " + ", ".join(f"{r.system}/{r.case_id}" for r in refs), ""]
    return lines


def render_markdown(summary: EvalSummary) -> str:
    """The Markdown report: every number is formatted from `summary`."""
    return "\n".join(
        [
            *_headline(summary),
            *_rates(summary),
            *_paired_lines(summary),
            *_cost_latency(summary),
            *_recovery_lines(summary),
            *_failures(summary),
            *_cases(summary),
            *_repeat_lines(summary),
            *_regression_lines(summary),
        ]
    )


__all__ = [
    "HARNESS_ERROR_CATEGORY",
    "EvalSummary",
    "PairedComparison",
    "SystemSummary",
    "build_summary",
    "render_markdown",
]
