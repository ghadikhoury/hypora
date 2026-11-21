"""Report the S5.1 protocol from S5.2's frozen records and publish recoverable copies.

Run this after collection, from the S5.2 reporting branch. The collection checkout remains
at eval-freeze. This script never invokes a model, runs a case, or writes a canonical record.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from hypora.cli.eval import load_cases
from hypora.core.schemas import ActorRole, EventType, ModelProfile, PolicyDecisionEvent, TaskType
from hypora.evaluation.case import Case, FaultKind
from hypora.evaluation.hidden_split import split_key
from hypora.evaluation.runner.records import CaseRecord
from hypora.evaluation.runner.store import GitRepo
from hypora.evaluation.system import CheckStatus
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from scripts import final_eval as final

TRAP_FAMILIES = {
    "target_leakage": "leakage",
    "grouped_rows": "grouped rows",
    "temporal_order": "temporal order",
    "class_imbalance": "class imbalance",
}
CONSTANT_CHOICES = {"existing_constant_trial", "constant_trial_validated"}
TRACE_TOOLS = {"constant_rule", "target_composition"}


def document_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialize in the frozen ResultsStore.write format."""
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def families(cases: Sequence[Case]) -> tuple[list[str], dict[str, list[str]]]:
    """Separate malformed safe stops and group the model-producing comparisons by trap."""
    malformed: list[str] = []
    groups: dict[str, list[str]] = {}
    for case in sorted(cases, key=lambda case: case.id):
        faults = {fault.kind for fault in case.injected_faults}
        if FaultKind.MALFORMED_DATA in faults:
            malformed.append(case.id)
            continue
        labels = {TRAP_FAMILIES[trap.kind] for trap in case.traps}
        if FaultKind.TOOL_ERROR in faults:
            labels.add("tool errors")
        for label in sorted(labels or {"clean controls"}):
            groups.setdefault(label, []).append(case.id)
    return malformed, dict(sorted(groups.items()))


def _decision_fields(reasons: Sequence[str], fields: set[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for reason in reasons:
        key, separator, value = reason.partition("=")
        if separator and key in fields:
            parsed[key] = json.loads(value)
    return parsed


def d060_view(
    events: Sequence[PolicyDecisionEvent], record: CaseRecord, *, regression: bool
) -> dict[str, Any]:
    """Extract only the frozen policy's selected action and flagged column subsets."""
    constant_decisions: list[dict[str, Any]] = []
    put_forward: set[str] = set()
    flagged: set[str] = set()
    for event in events:
        payload = event.payload
        if payload.tool == "constant_rule" and regression:
            decision = _decision_fields(
                payload.reasons, {"outcome", "action", "nominated", "chosen"}
            )
            constant_decisions.append(decision)
            if decision.get("action") in CONSTANT_CHOICES:
                chosen = decision.get("chosen")
                if not isinstance(chosen, str):
                    raise ValueError("constant rule named no trial for its selection")
                put_forward.add(chosen)
        if payload.tool == "target_composition":
            decision = _decision_fields(payload.reasons, {"outcome", "subset"})
            if decision.get("outcome") == "flagged":
                columns = decision.get("subset")
                if (
                    not isinstance(columns, list)
                    or not columns
                    or not all(isinstance(column, str) for column in columns)
                ):
                    raise ValueError("target-composition decision named no flagged columns")
                flagged.update(columns)
    model = record.outputs.final_model if record.outputs else None
    return {
        "constant_rule_decisions": constant_decisions,
        "constant_trials_put_forward": sorted(put_forward),
        "final_was_rule_constant": model is not None and model.trial_id in put_forward,
        "target_composition_columns": sorted(flagged),
    }


def trace_report(
    arms: Mapping[str, final.ArmRecords], cases: Mapping[str, Case], work_root: Path
) -> dict[str, Any]:
    """Report D-060 decisions per run, preserving unavailable traces as such."""
    rows: list[dict[str, Any]] = []
    for arm, records in arms.items():
        if not final.ARMS[arm].paid:
            continue
        for record in [
            *records.primary.values(),
            *(r for v in records.repeats.values() for r in v),
        ]:
            if not record.finished:
                continue
            database = work_root / "state" / f"{record.run_id}.sqlite"
            regression = cases[record.case_id].constraints.task_type is TaskType.REGRESSION
            run = {
                "arm": arm,
                "case_id": record.case_id,
                "run_id": record.run_id,
                "kind": record.kind.value,
                "attempt": record.attempt,
                "passed": record.passed,
                "regression": regression,
                "state_available": database.is_file(),
            }
            if not run["state_available"]:
                rows.append(run)
                continue
            db = Database(database)
            try:
                events = EventLog(db).read(
                    record.run_id,
                    EventFilters(
                        types=(EventType.POLICY_DECISION,), actor_role=ActorRole.CONTROLLER
                    ),
                )
            finally:
                db.close()
            decisions = [
                event
                for event in events
                if isinstance(event, PolicyDecisionEvent) and event.payload.tool in TRACE_TOOLS
            ]
            rows.append(run | d060_view(decisions, record, regression=regression))
    checked = [row for row in rows if row["state_available"]]
    return {
        "runs_checked": len(checked),
        "regression_runs_checked": sum(row["regression"] for row in checked),
        "state_unavailable": len(rows) - len(checked),
        "constant_rule_fired_runs": sum(
            any(decision.get("outcome") == "fired" for decision in row["constant_rule_decisions"])
            for row in checked
        ),
        "constant_put_forward_runs": sum(
            bool(row["constant_trials_put_forward"]) for row in checked
        ),
        "final_constant_runs": sum(row["final_was_rule_constant"] for row in checked),
        "composition_flagged_runs": sum(bool(row["target_composition_columns"]) for row in checked),
        "runs": rows,
    }


def build_report(commit: str, cases: Sequence[Case], collection: Path) -> dict[str, Any]:
    """Use the frozen primary, repeat and trace records to produce protocol-compliant tables."""
    cases_by_id = {case.id: case for case in cases}
    results_root = collection / "results"
    final_dir = results_root / "final"
    work_root = collection / "runs" / "final-s5.2" / "work"
    summary = final.build_final_summary(
        commit,
        list(cases_by_id),
        results_root=results_root,
        work_root=work_root,
        audit=final_dir / "s5.2-paid-accounting.json",
        orphans=final_dir / "s5.2-orphans",
    )
    arms = {name: final.load_arm(arm, commit, results_root) for name, arm in final.ARMS.items()}
    malformed, groups = families(cases)
    graded_case_ids = sorted(set(cases_by_id) - set(malformed))
    summary["comparison_case_count"] = len(graded_case_ids)
    summary["modelable_pass_rate"] = {
        arm: final.rate(
            [records.primary[case] for case in graded_case_ids if case in records.primary]
        )
        for arm, records in arms.items()
    }
    summary["malformed_safe_stops"] = [
        row for row in summary["per_case"] if row["case_id"] in malformed
    ]
    summary["modelable_paired"] = {}
    full_pairs = summary["paired"]
    summary["paired"] = {}
    for first, second in (("multi", "single"), ("multi", "baseline"), ("single", "baseline")):
        pair = f"{first}_vs_{second}"
        summary["paired"][pair] = full_pairs[pair]
        summary["modelable_paired"][pair] = {
            "first": first,
            "second": second,
            **final.paired(arms[first].primary, arms[second].primary, graded_case_ids),
        }
    summary["multi_vs_baseline_by_family"] = {
        family: {
            "case_ids": ids,
            **final.paired(arms["multi"].primary, arms["baseline"].primary, ids),
        }
        for family, ids in groups.items()
    }
    summary["d060"] = trace_report(arms, cases_by_id, work_root)
    summary["repeat_latency"] = repeat_latency(arms)
    summary["failure_analysis"] = failure_examples(arms, cases_by_id)
    return summary


def repeat_latency(arms: Mapping[str, final.ArmRecords]) -> list[dict[str, Any]]:
    """Per-case spread across graded repeat measurements, separate from the primary."""
    rows: list[dict[str, Any]] = []
    for arm, records in arms.items():
        for case in sorted(final.REPEAT_CASES):
            values = [
                record.timings.system_seconds
                for record in records.repeats.get(case, [])
                if record.finished
            ]
            rows.append(
                {
                    "arm": arm,
                    "case_id": case,
                    "repeat_count": len(values),
                    "minimum": min(values, default=None),
                    "median": median(values) if values else None,
                    "maximum": max(values, default=None),
                }
            )
    return rows


def failure_examples(
    arms: Mapping[str, final.ArmRecords], cases: Mapping[str, Case]
) -> list[dict[str, Any]]:
    """Select up to ten cases with failed checks or injected-fault evidence.

    Sort multi and single failures first. For each selected case retain the exact grader checks
    and their evidence refs for all failed arms. Fault scenarios can fill the sample when fewer
    than ten measured failures exist.
    """
    examples: list[dict[str, Any]] = []
    for case_id, case in sorted(cases.items()):
        failures = []
        for arm, records in arms.items():
            found = records.primary.get(case_id)
            if found is not None and found.finished and not found.passed:
                assert found.grade is not None
                failures.append(
                    {
                        "arm": arm,
                        "run_id": found.run_id,
                        "checks": [
                            check.model_dump(mode="json")
                            for check in found.grade.checks
                            if check.status is CheckStatus.FAILED
                        ],
                        "limits_fired": list(found.outputs.limits_fired) if found.outputs else [],
                    }
                )
        if failures or case.injected_faults:
            examples.append(
                {
                    "case_id": case_id,
                    "failures": failures,
                    "injected_faults": sorted({fault.kind.value for fault in case.injected_faults}),
                }
            )
    return sorted(
        examples,
        key=lambda row: (
            -sum(failure["arm"] in {"multi", "single"} for failure in row["failures"]),
            -len(row["failures"]),
            row["case_id"],
        ),
    )[:10]


def _counts(row: Mapping[str, Any]) -> str:
    if row["cases"] == 0:
        return "0 | | | | | | | | " + ", ".join(row["excluded_cases"])
    return (
        f"{row['cases']} | {row['both_pass']} | {row['first_only']} | {row['second_only']} "
        f"| {row['both_fail']} | {row['difference']:+.1%} "
        f"| {row['ci_low']:+.1%} to {row['ci_high']:+.1%} | {row['verdict']} "
        f"| {', '.join(row['excluded_cases']) or 'none'}"
    )


def render_report(summary: Mapping[str, Any]) -> str:
    """Render the frozen quality comparison, overall target counts and safety outcomes."""
    lines = [
        "Full-set primary rates and paired comparisons include every planned case. "
        "Multi vs single is the primary system comparison. The fixed pipeline is the simple "
        "automation reference: one default histogram-gradient-boosting trial, random splitting, "
        "all permitted non-target columns and no retries. All systems receive the same visible "
        "data and limits. Leakage, grouped, temporal and fault cases can penalize the fixed "
        "workflow by design. Interpret differences as workflow and methodology evidence, and "
        "use hidden metrics on comparable cases, including the clean controls, to assess "
        "predictive quality.",
        "",
        "## Modeling cases",
        "",
        f"{summary['comparison_case_count']} cases can produce a model. The two platform "
        "safe stops are reported separately below.",
        "",
        "| Arm | Passed | Graded | 95% Wilson interval |",
        "|---|---:|---:|---|",
    ]
    for arm, rate in summary["modelable_pass_rate"].items():
        interval = (
            f"{rate['wilson_low']:.1%} to {rate['wilson_high']:.1%}" if rate["graded"] else "n/a"
        )
        lines.append(f"| {arm} | {rate['passed']} | {rate['graded']} | {interval} |")
    lines += [
        "",
        "| Modeling cases | Paired | Both pass | First only | Second only | Both fail | "
        "Difference | 95% CI | Verdict | Excluded |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for pair in summary["modelable_paired"].values():
        lines.append(f"| {pair['first']} vs {pair['second']} | {_counts(pair)} |")
    lines += [
        "",
        "## Malformed-data safe stops",
        "",
        "| Case | Multi | Single | Fixed baseline |",
        "|---|---|---|---|",
    ]
    for row in summary["malformed_safe_stops"]:
        lines.append(
            f"| {row['case_id']} | {row['multi']['outcome']} | {row['single']['outcome']} "
            f"| {row['baseline']['outcome']} |"
        )
    lines += [
        "",
        "## Multi vs fixed baseline by family",
        "",
        "A case with two traps contributes to both relevant family rows. Family intervals are "
        "descriptive and can be imprecise on these small samples.",
        "",
        "| Family | Cases | Both pass | Multi only | Baseline only | Both fail | "
        "Difference | 95% CI | Verdict | Excluded |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    lines.extend(
        f"| {family} | {_counts(pair)} |"
        for family, pair in summary["multi_vs_baseline_by_family"].items()
    )
    lines += [
        "",
        "## Repeat latency spread",
        "",
        "Latency is the runner's system time, including in-run rate-limit and retry waits, "
        "excluding dataset setup and grading. Input cache use is provider-determined and is "
        "recorded with token counts and spend. The primary p95 uses nearest rank.",
        "",
        "| Arm | Case | Repeat runs graded | Min seconds | Median seconds | Max seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["repeat_latency"]:
        values = [
            f"{row[stat]:.0f}" if row[stat] is not None else "n/a"
            for stat in ("minimum", "median", "maximum")
        ]
        lines.append(
            f"| {row['arm']} | {row['case_id']} | {row['repeat_count']} | {' | '.join(values)} |"
        )
    lines += ["", "## Failure evidence", ""]
    for row in summary["failure_analysis"]:
        if row["failures"]:
            lines.append(f"- **{row['case_id']}**")
        else:
            kinds = ", ".join(row["injected_faults"])
            lines.append(f"- **{row['case_id']}** was designed to exercise {kinds}.")
        for failure in row["failures"]:
            checks = ", ".join(check["name"] for check in failure["checks"])
            limits = ", ".join(failure["limits_fired"]) or "none"
            lines.append(f"  - {failure['arm']} failed {checks}. Limits fired: {limits}.")
    lines += ["", "The JSON report retains the grader details and evidence references by run."]
    trace = summary["d060"]
    lines += [
        "",
        "## D-060 platform decisions",
        "",
        f"Checked {trace['runs_checked']} graded agent traces, including "
        f"{trace['regression_runs_checked']} regression traces. "
        f"State unavailable for {trace['state_unavailable']} runs. "
        f"The constant rule fired on {trace['constant_rule_fired_runs']} runs, "
        f"put a trial forward on {trace['constant_put_forward_runs']} runs, "
        f"and {trace['final_constant_runs']} kept it as the final model. "
        f"Target composition flagged columns on {trace['composition_flagged_runs']} runs.",
        "",
        "| Arm | Case | Run | Constant rule actions | Constant trials put forward | "
        "Constant finalized | Flagged columns |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in trace["runs"]:
        trials = row.get("constant_trials_put_forward", [])
        columns = row.get("target_composition_columns", [])
        actions = [
            decision["action"]
            for decision in row.get("constant_rule_decisions", [])
            if decision.get("outcome") == "fired"
        ]
        if actions or trials or columns:
            lines.append(
                f"| {row['arm']} | {row['case_id']} | {row['kind']} {row['attempt']} | "
                f"{', '.join(actions) or 'none'} | {', '.join(trials) or 'none'} | "
                f"{row['final_was_rule_constant']} | "
                f"{', '.join(columns) or 'none'} |"
            )
    base = final.render_final_markdown(summary)
    header, rest = base.split("\n\n", 1)
    return f"{header}\n\n{lines[0]}\n\n{rest}\n" + "\n".join(lines[1:]) + "\n"


def project_record(raw: bytes, case: Case) -> tuple[bytes, dict[str, Any]]:
    """Omit only the deterministic split directory key and verify byte-for-byte recovery."""
    document = json.loads(raw)
    record = CaseRecord.model_validate(document)
    if record.case_id != case.id or record.inputs.case_sha256 != case.content_hash():
        raise ValueError(f"record differs from frozen case {case.id}")
    split = document["inputs"]["split"]
    position = None
    if split is not None:
        position = list(split).index("split_key")
        del split["split_key"]
    projected = document_bytes(document)
    entry = {
        "case_id": case.id,
        "canonical_sha256": hashlib.sha256(raw).hexdigest(),
        "published_sha256": hashlib.sha256(projected).hexdigest(),
        "split_key_position": position,
    }
    if restore_record(projected, case, entry) != raw:
        raise ValueError(f"public projection could not reproduce the record for {case.id}")
    return projected, entry


def restore_record(projected: bytes, case: Case, entry: Mapping[str, Any]) -> bytes:
    """Recover the original bytes from a public copy and the frozen case."""
    if hashlib.sha256(projected).hexdigest() != entry["published_sha256"]:
        raise ValueError(f"published record SHA-256 mismatch for {case.id}")
    document = json.loads(projected)
    position = entry["split_key_position"]
    if position is not None:
        split = document["inputs"]["split"]
        fields = list(split.items())
        fields.insert(position, ("split_key", split_key(case, split["dataset_sha256"])))
        document["inputs"]["split"] = dict(fields)
    original = document_bytes(document)
    if hashlib.sha256(original).hexdigest() != entry["canonical_sha256"]:
        raise ValueError(f"canonical record SHA-256 mismatch for {case.id}")
    record = CaseRecord.model_validate_json(original)
    if record.case_id != case.id or record.inputs.case_sha256 != case.content_hash():
        raise ValueError(f"restored record differs from frozen case {case.id}")
    return original


def publish_records(
    results_root: Path, staging_root: Path, commit: str, cases: Mapping[str, Case]
) -> None:
    """Write public per-case records and an index linking to untouched canonical copies."""
    if results_root.resolve() == staging_root.resolve():
        raise ValueError("staging directory must differ from the canonical results directory")
    index: dict[str, Any] = {}
    for arm in final.ARMS.values():
        store = final.store_for(arm, commit, results_root)
        records = final.load_arm(arm, commit, results_root)
        entries = [
            *((store.path_for(record), record) for record in records.primary.values()),
            *(
                (store.path_for(record), record)
                for runs in records.repeats.values()
                for record in runs
            ),
            *((store.attempt_path(record), record) for record in records.attempts),
        ]
        for source, record in entries:
            projected, digest = project_record(source.read_bytes(), cases[record.case_id])
            relative = source.relative_to(results_root)
            destination = staging_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(projected)
            index[relative.as_posix()] = digest
    evidence = staging_root / "final"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "s5.2-evidence-index.json").write_bytes(
        document_bytes({"commit": commit, "records": dict(sorted(index.items()))})
    )


def _restore_all(staging_root: Path, destination: Path, cases: Mapping[str, Case]) -> None:
    if destination.exists():
        raise ValueError("restore needs a new directory")
    index = json.loads((staging_root / "final" / "s5.2-evidence-index.json").read_bytes())
    restored: dict[Path, bytes] = {}
    for path, entry in index["records"].items():
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != index["commit"]:
            raise ValueError("unsafe record path in evidence index")
        restored[relative] = restore_record(
            (staging_root / relative).read_bytes(), cases[entry["case_id"]], entry
        )
    for relative, raw in restored.items():
        out = destination / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)


def _record_slot(arm: final.Arm, record: CaseRecord) -> final.Slot:
    return final.Slot(arm.name, record.case_id, record.kind, record.attempt)


def validate_completed_matrix(
    results_root: Path, manifest: Mapping[str, Any], cases: Mapping[str, Case]
) -> None:
    """Require exactly the frozen slots and authenticate the provenance of every record."""
    commit = manifest["commit"]
    for name, arm in final.ARMS.items():
        config = manifest["configs"][arm.config]
        records = final.load_arm(arm, commit, results_root)
        active = [
            *records.primary.values(),
            *(record for repeats in records.repeats.values() for record in repeats),
        ]
        expected = {
            slot for which in final.Pass for slot in final.planned_slots(arm, which, list(cases))
        }
        observed = {_record_slot(arm, record) for record in active}
        missing = expected - observed
        unexpected = observed - expected
        ungraded = [_record_slot(arm, record) for record in active if not record.finished]
        if missing or unexpected or ungraded or len(active) != len(observed):
            problems = []
            for label, slots in (
                ("missing", missing),
                ("unexpected", unexpected),
                ("ungraded", ungraded),
            ):
                if slots:
                    details = sorted(f"{s.case_id}/{s.kind.value}/{s.attempt}" for s in slots)
                    problems.append(f"{label}: {', '.join(details)}")
            if len(active) != len(observed):
                problems.append("duplicate slots")
            raise ValueError(f"{name} matrix is incomplete or inconsistent: {'; '.join(problems)}")
        for record in records.attempts:
            if record.finished or _record_slot(arm, record) not in expected:
                raise ValueError(
                    f"{name}/{record.run_id} is not an archived failed collection attempt"
                )
        for record in [*active, *records.attempts]:
            identity = record.provenance
            case = cases[record.case_id]
            if (
                record.case_split is not case.split
                or record.inputs.case_sha256 != case.content_hash()
                or identity.dirty
                or identity.commit_sha != commit
                or identity.config != arm.config
                or identity.profile.value != config["profile"]
                or identity.flags.agent_mode.value != config["agent_mode"]
                or identity.run_config_hash != config["run_config_hash"]
                or identity.tiers.model_dump(mode="json") != config["tiers"]
            ):
                raise ValueError(
                    f"{name}/{record.run_id} provenance differs from the freeze manifest"
                )


def validate_accounting(collection: Path, audit_path: Path) -> None:
    """Require a complete paid audit that matches the original durable journal."""
    journal = collection / "runs" / "final-s5.2" / "paid-attempts.jsonl"
    if not audit_path.is_file() or not journal.is_file() or journal.stat().st_size == 0:
        raise ValueError("completed matrix needs its published paid audit and original journal")
    audit = json.loads(audit_path.read_bytes())
    state = final.SpendLedger(journal, cap_usd=final.CAP_USD, audit=audit_path).snapshot()
    if (
        audit
        != {
            "cap_usd": final.CAP_USD,
            "attempt_reservation_usd": final.ATTEMPT_RESERVATION_USD,
            "measured_usd": state.measured_usd,
            "unknown_reserved_usd": state.reserved_usd,
            "attempts": list(state.attempts),
        }
        or not state.attempts
        or state.measured_usd + state.reserved_usd > final.CAP_USD + 1e-9
    ):
        raise ValueError("paid audit differs from the journal or exceeds the frozen cap")


def publish_collection(collection: Path, staging: Path, cases: Sequence[Case]) -> None:
    """Validate complete local results, then install the public staging folder atomically."""
    source = collection / "results"
    audit_dir = source / "final"
    manifest = json.loads((audit_dir / "s5.2-manifest.json").read_bytes())
    commit = manifest["commit"]
    identity = final.read_identity(final.FREEZE_TAG, GitRepo(collection))
    final.check_identity(identity)
    if commit != identity.head or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("collection manifest commit differs from the freeze tag")
    lock = subprocess.run(
        ["git", "show", f"{identity.head}:results/readiness/lock.json"],
        cwd=collection,
        check=True,
        capture_output=True,
    ).stdout
    by_id = {case.id: case for case in cases}
    if (
        manifest["step"] != "S5.2"
        or manifest["freeze_tag"] != final.FREEZE_TAG
        or manifest["case_set"] != final.CASE_SET
        or manifest["case_set_sha256"] != final.case_set_sha256(cases)
        or manifest["cases"] != {case.id: case.content_hash() for case in cases}
        or manifest["repeat_cases"] != sorted(final.REPEAT_CASES)
        or manifest["repeats"] != final.REPEATS
        or manifest["readiness_lock_sha256"] != hashlib.sha256(lock).hexdigest()
        or manifest["cap_usd"] != final.CAP_USD
        or manifest["attempt_reservation_usd"] != final.ATTEMPT_RESERVATION_USD
        or set(manifest["configs"]) != {arm.config for arm in final.ARMS.values()}
        or any(
            config["profile"] != ModelProfile.EVAL.value or config["fallback_enabled"] is not False
            for config in manifest["configs"].values()
        )
    ):
        raise ValueError("collection manifest does not identify the supplied frozen protocol")
    validate_completed_matrix(source, manifest, by_id)
    validate_accounting(collection, audit_dir / "s5.2-paid-accounting.json")

    staging = staging.resolve()
    if staging.exists() or staging.is_relative_to(collection.resolve()):
        raise ValueError("stage into a new folder outside the collection checkout")
    summary = build_report(commit, cases, collection)
    staging.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging.parent, prefix=".s5.2-") as temp:
        temporary = Path(temp) / "results"
        publish_records(source, temporary, commit, by_id)
        summary_dir = temporary / "final"
        (summary_dir / "s5.2-summary.json").write_bytes(document_bytes(summary))
        (summary_dir / "s5.2-summary.md").write_text(render_report(summary), encoding="utf-8")
        for name in ("s5.2-manifest.json", "s5.2-paid-accounting.json"):
            shutil.copy2(audit_dir / name, summary_dir / name)
        if (audit_dir / "s5.2-orphans").is_dir():
            shutil.copytree(audit_dir / "s5.2-orphans", summary_dir / "s5.2-orphans")
        temporary.rename(staging)


def main() -> int:
    """Build reports or restore published records, with no access to the provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="frozen case directory")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report")
    report.add_argument("--collection", type=Path, required=True)
    report.add_argument("--staging-results", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--staging-results", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(str(args.cases))
    final.check_case_set(cases)
    by_id = {case.id: case for case in cases}
    if args.command == "restore":
        _restore_all(args.staging_results, args.output, by_id)
        return 0
    publish_collection(args.collection, args.staging_results, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
