#!/usr/bin/env python3
"""Generate F3 (Multi-agent coordination) evidence for S5.3.

Reuses S5.2 frozen run records (`results/final/s5.2-summary.json` and SQLite event logs under
`/home/ghadi/hypora-final/runs/final-s5.2/work/state/`) without duplicating paid S5.2 runs, and
executes 4 deterministic agent-failure tests against the real `RunController` in temp storage.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]

from hypora.cli.trace import trace_rows
from hypora.core.config import ModelsConfig, load_settings
from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.contract import build_data_contract
from hypora.core.controller.controller import RunController, RunRequest
from hypora.core.llm import (
    LLMAdapter,
    ProviderAdmission,
    ProviderError,
    ProviderErrorKind,
    ProviderRequest,
    ProviderResult,
    load_price_table,
)
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.types import MessageRole, ToolCall
from hypora.core.schemas import ApprovalPolicy, Provider, TaskSpecification
from hypora.core.schemas.events import ErrorEvent, HandoffEvent, ModelCallEvent, RoleStartedEvent
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_S52_RESULTS = REPO_ROOT / "results"
DEFAULT_RUNS_STATE = Path("/home/ghadi/hypora-final/runs/final-s5.2/work/state")
FREEZE_SHA = "62b4e3fb6016a0abddf70a14152252a62c8aca4a"
HARDWARE = DetectedResources(cpu_cores=4, total_memory_gb=8.0, gpu_detection=GpuDetection.NOT_FOUND)

DISCORDANT_CASES: tuple[tuple[str, str, str, str], ...] = (
    ("helped", "test_fault_malformed_landsat", "run_3983f7c5c2b6", "run_897e438a300e"),
    ("helped", "test_fault_malformed_maternal", "run_aa54d7c736e1", "run_11272ab5e2d6"),
    ("helped", "test_imb_bankruptcy", "run_064d9fd16ea6", "run_6ce71aadc88c"),
    ("hurt", "test_fault_submit_superconductivity", "run_90f177e2ffaf", "run_7d97a6e97e79"),
)

HANDOFF_EXAMPLE = ("test_clean_rice", "run_2a2d5305864e")


def _unthrottled(models: ModelsConfig) -> ModelsConfig:
    data = models.model_dump(mode="json")
    data["rate_limits"]["limits"] = []
    return ModelsConfig.model_validate(data)


def _strategy_report(run_id: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    fields = (
        "task_fit",
        "evidence_quality",
        "privacy",
        "license",
        "hardware_feasibility",
        "setup_cost",
        "expected_inference_cost",
        "latency_uncertainty",
        "evaluation_risk",
    )
    rows = {name: {"rating": "unknown", "note": "Unavailable"} for name in fields}
    paths = ("hosted_api", "pretrained_as_is", "fine_tune", "task_specific_training")
    options = [
        {
            "option_id": f"option_{i}",
            "path": path,
            "candidate": path,
            "summary": "Unknown",
            "tradeoffs": rows,
        }
        for i, path in enumerate(paths)
    ]
    citations: list[dict[str, Any]] = []
    if evidence is not None:
        ref = str(evidence["ref"])
        for opt in options:
            opt["evidence_refs"] = [ref]
        citations.append(
            {
                "ref": ref,
                "source_url": evidence["source_url"],
                "revision": evidence["revision"],
                "retrieved_on": str(evidence["retrieved_on"])[:10],
            }
        )
    return {
        "run_id": run_id,
        "options": options,
        "citations": citations,
        "recommendation": {"kind": "no_model_needed", "rationale": "Insufficient evidence"},
        "confidence": 0.0,
        "open_questions": ["Evidence is unavailable."],
    }


class _FailureScript:
    """Scripted provider driving A/B/C through one targeted role failure."""

    UNCITED_CLAIM = "Guaranteed private provider with zero evidence"

    def __init__(self, run_id: str, scenario: str) -> None:
        self.run_id = run_id
        self.scenario = scenario
        self.evidence: dict[str, Any] | None = None
        self.tool_calls: list[ToolCall] = []
        self.c_prompts: list[str] = []

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        sys_prompt = request.messages[0].content
        schema = request.output_schema_name

        if schema == "IntakeDraft":
            return json_result(
                {
                    "objective": "Predict whether a breast tumour is malignant",
                    "target": "target",
                    "task_type": "binary_classification",
                    "primary_metric": "roc_auc",
                }
            )
        if schema == "PlanDraft":
            return json_result(
                {
                    "steps": [
                        {
                            "tool": None,
                            "purpose": "Inspect permitted evidence",
                            "success_check": "Reviewed",
                        }
                    ]
                }
            )
        if schema == "ReportNarrative":
            return json_result(
                {
                    "summary": "Run stopped safely after a role failure.",
                    "observations": ["Failure recorded cleanly."],
                    "limitations": ["No model trained."],
                }
            )
        if schema == "FinalAnswer":
            return json_result({"final_trial_id": None, "confidence": 0.0})
        if schema == "WorkerProposal":
            return json_result({"workers": []})
        if schema == "LeadAction":
            if self.scenario == "agent_a_refused_assignment":
                return json_result(
                    {
                        "action": "continue",
                        "reason": "Proceed",
                        "assignment": "import pandas as pd; df = pd.read_csv('secret.csv')",
                    }
                )
            resp: dict[str, Any] = {"action": "continue", "reason": "Proceed within scope"}
            if "assignment field" in request.messages[1].content:
                resp["assignment"] = "Inspect approved data and run a bounded baseline"
            return json_result(resp)
        if schema == "ResearchObservation":
            if self.scenario == "agent_b_uncited_claim":
                bad = _strategy_report(self.run_id, evidence=None)
                bad["recommendation"] = {
                    "kind": "task_specific_training",
                    "rationale": self.UNCITED_CLAIM,
                }
                return json_result(
                    {
                        "summary": "Uncited claim",
                        "uncertainties": [],
                        "strategy_report": bad,
                    }
                )
            return json_result(
                {
                    "summary": "Research reviewed",
                    "uncertainties": [],
                    "strategy_report": _strategy_report(self.run_id, self.evidence),
                }
            )
        if schema is None and "You are Agent B" in sys_prompt:
            if self.scenario == "agent_b_provider_failure":
                raise ProviderError(
                    ProviderErrorKind.INVALID_REQUEST,
                    "Simulated Agent B provider outage",
                )
            last = request.messages[-1]
            if last.role is not MessageRole.TOOL:
                call = ToolCall(
                    id="b_ret_1",
                    name="retrieve_methodology",
                    arguments={"topic": "evaluation"},
                )
                self.tool_calls.append(call)
                return tool_call_result(call)
            content = json.loads(last.content)
            self.evidence = content["passages"][0]
            return text_result("Research reviewed.")
        if "You are Hypora's data and implementation specialist" in sys_prompt:
            if schema == "DataContract":
                context = request.messages[1].content
                decoder = json.JSONDecoder()
                task = decoder.raw_decode(context.split("Task specification: ", 1)[1])[0]
                profile = decoder.raw_decode(context.split("Target-free dataset metadata: ", 1)[1])[
                    0
                ]
                spec = TaskSpecification.model_validate(task)
                columns = [item["name"] for item in profile["tabular"]["columns"]]
                columns.append(spec.target)
                return json_result(build_data_contract(spec, columns).model_dump(mode="json"))
            return text_result("Metadata reviewed.")
        if schema is None and "You are Agent C" in sys_prompt:
            self.c_prompts.append("\n".join(m.content for m in request.messages))
            if self.scenario == "agent_c_execution_failure":
                raise ProviderError(
                    ProviderErrorKind.INVALID_REQUEST,
                    "Simulated Agent C coding failure",
                )
            return text_result("No trial submitted.")
        return text_result("Done.")


class _NoWorker:
    def __init__(
        self,
        database: Path,
        runs_root: Path,
        log: Path,
        *,
        worker_id: str = "",
        poll_seconds: float = 0.0,
    ) -> None:
        pass

    def start(self) -> None:
        return None

    def alive(self) -> bool:
        return False

    def stop(self) -> int | None:
        return 0


async def _run_failure_scenario(scenario: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"f3_{scenario}_") as tmp:
        root = Path(tmp)
        data_path = root / "breast_cancer.csv"
        load_breast_cancer(as_frame=True).frame.to_csv(data_path, index=False)
        db = Database(root / "hypora.sqlite")
        migrate(db)
        try:
            settings = load_settings()
            base_cfg = RunConfig.build(settings, hardware=HARDWARE)
            flags = base_cfg.flags.model_copy(
                update={
                    "agent_mode": AgentMode.MULTI,
                    "approval_policy": ApprovalPolicy.AUTO_WITHIN_CONTRACT,
                    "planning": Toggle.ON,
                }
            )
            run_config = base_cfg.model_copy(update={"flags": flags})
            models = _unthrottled(run_config.models)
            run_id = f"run_f3_{scenario}"
            script = _FailureScript(run_id, scenario)
            adapter = LLMAdapter(
                models=models,
                profile=run_config.model_profile,
                prices=load_price_table(),
                clients={Provider.GEMINI: script},
                events=EventLog(db),
                admission=ProviderAdmission(models.rate_limits),
            )
            controller = RunController(
                database=db,
                runs_root=root / "runs",
                run_config=run_config,
                adapter=adapter,
                poll_seconds=0.1,
                worker_factory=_NoWorker,
            )
            result = await controller.run(
                RunRequest(dataset=data_path, goal="Predict malignancy", run_id=run_id)
            )
            events = list(EventLog(db).read(run_id))
            errors = [
                {
                    "seq": e.seq,
                    "role": e.actor_role.value,
                    "code": e.payload.code,
                    "message": e.payload.message,
                }
                for e in events
                if isinstance(e, ErrorEvent)
            ]
            handoffs = [
                {
                    "seq": e.seq,
                    "from": e.payload.handoff.from_role.value,
                    "to": e.payload.handoff.to.value,
                    "status": e.payload.handoff.status.value,
                }
                for e in events
                if isinstance(e, HandoffEvent)
            ]
            roles = [e.actor_role.value for e in events if isinstance(e, RoleStartedEvent)]
            failed_model_calls = [
                {
                    "seq": e.seq,
                    "role": e.actor_role.value,
                    "error_code": e.payload.error_code,
                    "status": e.payload.status.value,
                }
                for e in events
                if isinstance(e, ModelCallEvent) and e.payload.status.value == "error"
            ]
            if scenario == "agent_b_provider_failure":
                passed = (
                    "agent_b" in roles
                    and any(c["role"] == "agent_b" for c in failed_model_calls)
                    and any(
                        h["from"] == "agent_b" and h["status"] in {"failed", "partial"}
                        for h in handoffs
                    )
                    and result.state.status.value == "FAILED"
                    and result.final_trial_id is None
                    and not result.verified
                    and result.report_md.exists()
                )
            elif scenario == "agent_b_uncited_claim":
                uncited_forwarded = any(_FailureScript.UNCITED_CLAIM in p for p in script.c_prompts)
                passed = (
                    "agent_b" in roles
                    and result.report_md.exists()
                    and not result.verified
                    and not uncited_forwarded
                )
            elif scenario == "agent_a_refused_assignment":
                passed = (
                    "agent_a" in roles
                    and "agent_c" not in roles
                    and result.state.status.value == "FAILED"
                    and result.final_trial_id is None
                    and not result.verified
                    and result.report_md.exists()
                    and len(errors) >= 1
                )
            else:
                passed = (
                    "agent_b" in roles
                    and "agent_c" in roles
                    and any(c["role"] == "agent_c" for c in failed_model_calls)
                    and any(h["from"] == "agent_c" and h["status"] == "failed" for h in handoffs)
                    and result.state.status.value == "FAILED"
                    and result.final_trial_id is None
                    and not result.verified
                    and result.report_md.exists()
                )
            return {
                "scenario": scenario,
                "passed": passed,
                "final_status": result.state.status.value,
                "final_model_is_none": result.final_trial_id is None,
                "verified": result.verified,
                "report_exists": result.report_md.exists(),
                "event_count": len(events),
                "roles_started": roles,
                "error_events": errors,
                "failed_model_calls": failed_model_calls,
                "handoffs": handoffs,
                "notes": list(result.notes),
            }
        finally:
            db.close()


def _summarize_run_trace(
    case_id: str, config_dir: str, run_id: str, results_root: Path, state_dir: Path
) -> dict[str, Any]:
    rec_path = results_root / FREEZE_SHA / config_dir / "agent" / f"{case_id}.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    failed_checks = [
        {"name": c["name"], "detail": c["detail"]}
        for c in rec.get("grade", {}).get("checks", [])
        if c["status"] == "failed"
    ]
    db_path = state_dir / f"{run_id}.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"Missing S5.2 state DB: {db_path}")
    db = Database(db_path)
    try:
        events = list(EventLog(db).read(run_id))
    finally:
        db.close()
    rows = trace_rows(events)
    timeline = [
        {
            "seq": r.seq,
            "offset_s": round(r.offset_s, 1),
            "role": r.role,
            "type": r.type,
            "detail": r.detail,
        }
        for r in rows
        if r.type in {"stage_transition", "role_started", "handoff", "error", "job_transition"}
        or "error" in r.detail.lower()
    ]
    return {
        "run_id": run_id,
        "config": config_dir,
        "passed": bool(rec["passed"]),
        "decision_class": rec.get("outputs", {}).get("decision_class"),
        "failed_checks": failed_checks,
        "wall_seconds": round(float(rec.get("timings", {}).get("system_seconds", 0.0)), 2),
        "actual_spend_usd": round(float(rec.get("usage", {}).get("actual_spend_usd", 0.0)), 6),
        "model_calls": int(rec.get("usage", {}).get("model_calls", 0)),
        "tool_calls": int(rec.get("usage", {}).get("tool_calls", 0)),
        "total_events": len(events),
        "roles_started": [e.actor_role.value for e in events if isinstance(e, RoleStartedEvent)],
        "error_events": [
            {
                "seq": e.seq,
                "role": e.actor_role.value,
                "code": e.payload.code,
                "message": e.payload.message[:200],
            }
            for e in events
            if isinstance(e, ErrorEvent)
        ],
        "key_timeline": timeline,
    }


def _build_handoff_trace(results_root: Path, state_dir: Path) -> dict[str, Any]:
    case_id, run_id = HANDOFF_EXAMPLE
    summary = _summarize_run_trace(case_id, "default", run_id, results_root, state_dir)
    handoff_rows = [
        r
        for r in summary["key_timeline"]
        if r["type"] in {"stage_transition", "role_started", "handoff"}
    ]
    return {
        "case_id": case_id,
        "run_id": run_id,
        "passed": summary["passed"],
        "total_events": summary["total_events"],
        "roles_started": summary["roles_started"],
        "handoff_timeline": handoff_rows[:40],
    }


def _case_explanation(
    kind: str, case_id: str, multi: dict[str, Any], single: dict[str, Any]
) -> str:
    if kind == "helped" and case_id.startswith("test_fault_malformed_"):
        return (
            "Both multi and single hit `CONTRACT: malformed data (TARGET_MISSING)` and stopped "
            "safely without training (`fault_malformed_data_0` passed on both). However, single "
            "emitted `decision_class='infeasible'` instead of `'train'`, failing `decision_class` "
            "and `final_model_provenance`, whereas multi preserved `decision_class='train'` "
            "through Agent A's intake/planning flow and passed."
        )
    if case_id == "test_imb_bankruptcy":
        return (
            f"Single failed in CONTRACT at {single['wall_seconds']}s (`MethodologyError: every "
            "non-target column must be accounted for`), never reaching Phase B. Multi iterated "
            "between Agent C and Agent A (even recovering from an `UNKNOWN_COLUMN` diagnostic "
            "error), completed a valid data contract, ran 4 trials + 2 bounded workers, and "
            "passed verification."
        )
    return (
        "Hurt case: both multi and single recovered from the injected `RATE_LIMITED` fault on "
        "`submit_experiment`. Single selected a candidate whose final `stage 2 (fit)` finished in "
        "~242s (passing), whereas multi proposed a heavier candidate whose final `stage 2 (fit)` "
        "hit the 300.0s per-job timeout (`FAILED_TERMINAL`), failing the run while also spending "
        "more time and calls."
    )


def generate_f3_evidence(
    results_root: Path = DEFAULT_S52_RESULTS, state_dir: Path = DEFAULT_RUNS_STATE
) -> dict[str, Any]:
    """Build complete F3 multi-agent evidence dictionary from frozen S5.2 + failure tests."""
    s52 = json.loads((results_root / "final" / "s5.2-summary.json").read_text(encoding="utf-8"))
    failure_scenarios = [
        asyncio.run(_run_failure_scenario(name))
        for name in (
            "agent_b_provider_failure",
            "agent_b_uncited_claim",
            "agent_a_refused_assignment",
            "agent_c_execution_failure",
        )
    ]
    handoff_trace = _build_handoff_trace(results_root, state_dir)
    case_comparisons = []
    for kind, case_id, multi_run, single_run in DISCORDANT_CASES:
        multi = _summarize_run_trace(case_id, "default", multi_run, results_root, state_dir)
        single = _summarize_run_trace(case_id, "single_agent", single_run, results_root, state_dir)
        case_comparisons.append(
            {
                "category": kind,
                "case_id": case_id,
                "multi": multi,
                "single": single,
                "explanation": _case_explanation(kind, case_id, multi, single),
            }
        )

    return {
        "feature": "F3",
        "title": "Multi-agent coordination",
        "freeze_commit": FREEZE_SHA,
        "core_comparison": {
            "source": "results/final/s5.2-summary.json",
            "primary_pass_rates": {
                arm: s52["systems"][arm]["pass_rate"] for arm in ("multi", "single")
            },
            "paired_multi_vs_single": s52["paired"]["multi_vs_single"],
            "cost_and_latency": {
                arm: {
                    "latency_seconds": s52["systems"][arm]["latency_seconds"],
                    "cost_per_graded_task": s52["systems"][arm]["cost_per_graded_task"],
                    "spend_all_runs": s52["systems"][arm]["spend_all_runs"],
                }
                for arm in ("multi", "single")
            },
        },
        "working_handoff_trace": handoff_trace,
        "agent_failure_tests": failure_scenarios,
        "helped_hurt_cases": case_comparisons,
    }


def render_f3_markdown(evidence: dict[str, Any]) -> str:
    """Render concise human-readable F3 evidence report."""
    core = evidence["core_comparison"]
    multi_pr = core["primary_pass_rates"]["multi"]
    single_pr = core["primary_pass_rates"]["single"]
    paired = core["paired_multi_vs_single"]
    multi_lat = core["cost_and_latency"]["multi"]["latency_seconds"]
    single_lat = core["cost_and_latency"]["single"]["latency_seconds"]
    multi_cost = core["cost_and_latency"]["multi"]["cost_per_graded_task"]
    single_cost = core["cost_and_latency"]["single"]["cost_per_graded_task"]
    multi_all = core["cost_and_latency"]["multi"]["spend_all_runs"]
    single_all = core["cost_and_latency"]["single"]["spend_all_runs"]

    lines = [
        "# S5.3 F3 Evidence - Multi-agent coordination",
        "",
        f"Frozen evaluation commit: `{evidence['freeze_commit']}` "
        "(`results/final/s5.2-summary.json`).",
        "",
        "## 1. Full 26-case comparison (Evaluation, 4 pts)",
        "",
        "| System | Primary Pass | 95% Wilson | Mean Latency | p95 Latency | "
        "Actual $/task | All-run $ |",
        "|---|---:|---|---:|---:|---:|---:|",
        f"| `multi` (A/B/C) | {multi_pr['passed']} / {multi_pr['graded']} "
        f"({multi_pr['rate'] * 100:.1f}%) | "
        f"{multi_pr['wilson_low'] * 100:.1f}% to {multi_pr['wilson_high'] * 100:.1f}% | "
        f"{multi_lat['mean']:.0f} s | {multi_lat['p95']:.0f} s | "
        f"${multi_cost['actual_usd']:.4f} | ${multi_all['actual_usd']:.4f} |",
        f"| `single` (`single_agent`) | {single_pr['passed']} / {single_pr['graded']} "
        f"({single_pr['rate'] * 100:.1f}%) | "
        f"{single_pr['wilson_low'] * 100:.1f}% to {single_pr['wilson_high'] * 100:.1f}% | "
        f"{single_lat['mean']:.0f} s | {single_lat['p95']:.0f} s | "
        f"${single_cost['actual_usd']:.4f} | ${single_all['actual_usd']:.4f} |",
        "",
        f"- **Paired bootstrap (10,000 resamples, seed 1990):** diff "
        f"`{paired['difference']:+.3f}` ({paired['ci_low']:+.3f} to {paired['ci_high']:+.3f}), "
        f"verdict `{paired['verdict']}` (both pass {paired['both_pass']}, "
        f"multi-only {paired['first_only']}, single-only {paired['second_only']}, "
        f"both fail {paired['both_fail']}).",
        "",
        "## 2. Working A -> B -> A -> C -> A handoff trace (Implementation, 3 pts)",
        "",
        f"Example: `{evidence['working_handoff_trace']['case_id']}` "
        f"(`{evidence['working_handoff_trace']['run_id']}`, "
        f"`passed={evidence['working_handoff_trace']['passed']}`, "
        f"`{evidence['working_handoff_trace']['total_events']}` total events).",
        "",
        "| Seq | +Time | Role | Event | Detail |",
        "|---:|---:|---|---|---|",
    ]
    for row in evidence["working_handoff_trace"]["handoff_timeline"][:18]:
        detail = row["detail"][:90]
        lines.append(
            f"| {row['seq']} | {row['offset_s']}s | `{row['role']}` | `{row['type']}` | {detail} |"
        )

    lines.extend(["", "## 3. Agent-failure tests (Architecture, 1 pt)", ""])
    lines.append("| Scenario | Passed | Final Status | Final Model | Events | Errors |")
    lines.append("|---|---|---|---|---:|---:|")
    for s in evidence["agent_failure_tests"]:
        model_state = "None" if s["final_model_is_none"] else "present"
        lines.append(
            f"| `{s['scenario']}` | `{s['passed']}` | `{s['final_status']}` | "
            f"{model_state} | {s['event_count']} | {len(s['error_events'])} |"
        )

    lines.extend(["", "## 4. Helped / hurt coordination cases (Analysis, 2 pts)", ""])
    for item in evidence["helped_hurt_cases"]:
        m, sg = item["multi"], item["single"]
        lines.extend(
            [
                f"### {item['category'].upper()}: `{item['case_id']}`",
                f"- **multi (`{m['run_id']}`):** `passed={m['passed']}`, "
                f"{m['wall_seconds']}s, ${m['actual_spend_usd']:.4f}, "
                f"errors={len(m['error_events'])}, failed_checks={m['failed_checks']}",
                f"- **single (`{sg['run_id']}`):** `passed={sg['passed']}`, "
                f"{sg['wall_seconds']}s, ${sg['actual_spend_usd']:.4f}, "
                f"errors={len(sg['error_events'])}, failed_checks={sg['failed_checks']}",
                f"- **Analysis:** {item['explanation']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    out_dir = REPO_ROOT / "results" / "s5.3" / "f3"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = generate_f3_evidence()
    (out_dir / "f3-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "f3-summary.md").write_text(render_f3_markdown(evidence), encoding="utf-8")
    print(f"Wrote {out_dir / 'f3-evidence.json'} and {out_dir / 'f3-summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
