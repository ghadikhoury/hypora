#!/usr/bin/env python3
"""Generate F5 (MCP Server) evaluation evidence for S5.3.

Evaluates the real stdio MCP server (`python -m hypora.mcp_server`) across:
  1. Dual-client verification (`mcp.Client` high-level SDK + low-level
     `stdio_client`/`ClientSession`), session-secret authentication (0600 hash-only file,
     missing/wrong secret rejected), and actionable `ToolError` validation recovery checks.
  2. 24 distinct typed tool-calling tasks (`MCP-01`..`MCP-24`) across TWO live models:
       - `openai/gpt-6-luna` (`ModelProfile.EVAL`, paid tier, hard $1.00 spend cap)
       - `gemini/gemini-3.5-flash-lite` (`ModelProfile.DEV`, free tier, $0 actual spend)
  3. Tool-description revision comparison: `v0_minimal` (terse 1-line descriptions, stripped
     property hints) vs `v1_production` (Hypora's production §12 structured descriptions from
     `docs/mcp/tools.json`, unmodified), plus a 6x3 repeat sample on both models and versions.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from hypora.core.config import load_settings
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.config.run_config import RunConfig
from hypora.core.llm import LLMAdapter, ProviderAdmission
from hypora.core.llm.errors import LLMCallError
from hypora.core.llm.types import CallContext, Message, ToolSpec
from hypora.core.schemas import ActorRole, EventType, ModelProfile, ModelTier
from hypora.core.schemas.trials import TrialDesign
from hypora.evaluation.stats.core import (
    cost_per_task,
    latency_summary,
    paired_bootstrap_difference,
    paired_outcome_table,
    wilson_interval,
)
from hypora.execution.protocol import experiments as protocol
from hypora.mcp_server.dev import create_dev_workspace, dev_secret
from hypora.mcp_server.session import SESSION_SECRET_ENV, Session, SessionConfig
from hypora.mcp_server.tools import HyporaTools
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.jobs import WorkerCapacity

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO_ROOT / "hypora" / "evaluation" / "cases" / "mcp" / "tasks.json"
OUT_DIR = REPO_ROOT / "results" / "s5.3" / "f5"
CAP_USD = 1.00
PRICE_RETRIEVED_ON = "2026-11-07"
HARDWARE = DetectedResources(cpu_cores=4, total_memory_gb=8.0, gpu_detection=GpuDetection.NOT_FOUND)
PIPELINE_SAMPLE = (
    REPO_ROOT / "hypora" / "execution" / "templates" / "tabular" / "linear" / "pipeline.py"
).read_text(encoding="utf-8")
DESIGN_SAMPLE: dict[str, Any] = {
    "kind": "baseline",
    "operation": "fit",
    "hypothesis": "Regularized logistic regression baseline for F5 MCP evaluation",
    "model_family": "logistic_regression",
    "expected_effect": "roc_auc above 0.90",
    "estimated_cost": {"compute_seconds": 15},
}
MINIMAL_DESCRIPTIONS: dict[str, str] = {
    "search_model_catalog": "Search the model catalog.",
    "get_model_evidence": "Get model evidence.",
    "retrieve_methodology": "Retrieve methodology passages.",
    "inspect_dataset": "Inspect a dataset.",
    "run_diagnostic": "Run a dataset diagnostic.",
    "write_trial_files": "Write trial files.",
    "submit_experiment": "Submit an experiment.",
    "get_experiment_status": "Get experiment status.",
    "get_experiment_metrics": "Get experiment metrics.",
    "get_experiment_logs": "Get experiment logs.",
}
OFFLINE_TOOLS: frozenset[str] = frozenset(
    {
        "inspect_dataset",
        "run_diagnostic",
        "retrieve_methodology",
        "write_trial_files",
        "submit_experiment",
        "get_experiment_status",
        "get_experiment_metrics",
        "get_experiment_logs",
    }
)


class _MemorySink:
    """Minimal EventSink for standalone LLMAdapter calls."""

    def __init__(self) -> None:
        self.events: list[tuple[EventType | str, Any]] = []

    def append(
        self,
        run_id: str,
        type: EventType | str,
        actor_role: ActorRole | str,
        payload: object,
        **_: Any,
    ) -> int:
        self.events.append((type, payload))
        return len(self.events)


def load_tasks(path: Path = TASKS_PATH) -> dict[str, Any]:
    """Load the 24-task F5 MCP benchmark."""
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _stdio_params(session_file: Path, secret: str | None = None) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "hypora.mcp_server", "--session", str(session_file)],
        cwd=str(REPO_ROOT),
        env={SESSION_SECRET_ENV: dev_secret(session_file) if secret is None else secret},
    )


def _strip_schema_descriptions(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_schema_descriptions(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_strip_schema_descriptions(item) for item in node]
    return node


def strip_tool_descriptions(tools: Sequence[ToolSpec]) -> list[ToolSpec]:
    """Create `v0_minimal` tool specs from production `v1_production` specs."""
    stripped: list[ToolSpec] = []
    for tool in tools:
        schema = _strip_schema_descriptions(copy.deepcopy(tool.input_schema))
        stripped.append(
            ToolSpec(
                name=tool.name,
                description=MINIMAL_DESCRIPTIONS.get(tool.name, f"Call {tool.name}."),
                input_schema=schema,
            )
        )
    return stripped


def _seed_workspace(ws_dir: Path) -> tuple[Path, str, str, str]:
    """Create a fresh dev workspace with one seeded trial and one completed job."""
    session_file = create_dev_workspace(ws_dir, force=True)
    config = SessionConfig.model_validate_json(session_file.read_bytes())
    config = config.model_copy(update={"status_wait_seconds": 0.0})
    session_file.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")

    session = Session(config)
    try:
        tools = HyporaTools(session)
        written = tools.write_trial_files(
            design=TrialDesign.model_validate(DESIGN_SAMPLE),
            pipeline_py=PIPELINE_SAMPLE,
            idempotency_key=f"{session.run_id}:seed_w_{uuid.uuid4().hex[:8]}",
        )
        submitted = tools.submit_experiment(
            trial_id=written.trial_id,
            idempotency_key=f"{session.run_id}:seed_s_{uuid.uuid4().hex[:8]}",
        )
        job_id = submitted.job_id
        job = session.queue.lease("f5-worker", WorkerCapacity(8, 0, 64.0))
        assert job is not None and job.id == job_id
        session.queue.start(job_id, "f5-worker")
        folder = protocol.job_dir(session.runs_root, session.run_id, job_id)
        log_lines = [f"stage0 line {i}: metric_step={0.80 + 0.01 * i:.2f}" for i in range(1, 11)]
        protocol.append_stage_log(
            folder,
            protocol.StageLog(
                attempt=1,
                stage=0,
                action="fit",
                status="completed",
                duration_s=0.4,
                lines=log_lines,
            ),
        )
        scores_payload: dict[str, Any] = {
            "folds": [
                {"fold": 0, "metrics": {"roc_auc": 0.921}},
                {"fold": 1, "metrics": {"roc_auc": 0.927}},
            ]
        }
        metrics_bytes = json.dumps(scores_payload).encode("utf-8")
        stored = ArtifactStore(
            session.database, session.runs_root / session.run_id / "artifacts"
        ).write(f"{job_id}.metrics", written.trial_id, "metrics", [metrics_bytes])
        protocol.write_json_once(
            folder / protocol.OUTCOME_FILE,
            protocol.ExperimentOutcome(
                job_id=job_id,
                succeeded=True,
                artifacts=[
                    protocol.OutcomeArtifact(
                        artifact_id=stored.id,
                        kind="metrics",
                        sha256=stored.sha256,
                        bytes=len(metrics_bytes),
                    )
                ],
                scores=scores_payload,
            ),
        )
        session.queue.complete(job_id, "f5-worker")
        return session_file, session.run_id, written.trial_id, job_id
    finally:
        session.close()


async def verify_stdio_auth_and_clients() -> dict[str, Any]:
    """Verify two real stdio MCP clients, session auth, and actionable ToolError recovery."""
    with tempfile.TemporaryDirectory(prefix="f5_auth_") as tmp:
        session_file, run_id, trial_id, job_id = _seed_workspace(Path(tmp) / "ws")
        secret = dev_secret(session_file)
        file_mode = stat.S_IMODE(session_file.stat().st_mode)
        raw_session_text = session_file.read_text(encoding="utf-8")
        secret_hidden = secret not in raw_session_text

        # Auth failure checks: missing secret and wrong secret exit non-zero before serving
        env_missing = {k: v for k, v in os.environ.items() if k != SESSION_SECRET_ENV}
        proc_missing = subprocess.run(
            [sys.executable, "-m", "hypora.mcp_server", "--session", str(session_file)],
            cwd=REPO_ROOT,
            env=env_missing,
            input="",
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        env_wrong = {**env_missing, SESSION_SECRET_ENV: "wrong-secret-value"}
        proc_wrong = subprocess.run(
            [sys.executable, "-m", "hypora.mcp_server", "--session", str(session_file)],
            cwd=REPO_ROOT,
            env=env_wrong,
            input="",
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

        # Client 1: high-level mcp.Client over stdio
        async with Client(_stdio_params(session_file), read_timeout_seconds=60) as client1:
            listed1 = await client1.list_tools()
            c1_tools = [t.name for t in listed1.tools]
            c1_inspect = await client1.call_tool("inspect_dataset", {})
            # Actionable error checks
            err_unknown = await client1.call_tool("inspect_dataset", {"bogus_arg": 1})
            err_idem = await client1.call_tool(
                "submit_experiment", {"trial_id": trial_id, "idempotency_key": "bad_key"}
            )
            err_missing = await client1.call_tool(
                "submit_experiment",
                {"trial_id": "tr_nonexistent", "idempotency_key": f"{run_id}:miss_1"},
            )
            err_diag = await client1.call_tool(
                "run_diagnostic",
                {"diagnostic": "feature_predictiveness", "column": "mean radius"},
            )
            recovered_diag = await client1.call_tool(
                "run_diagnostic",
                {
                    "diagnostic": "feature_predictiveness",
                    "column": "mean radius",
                    "target": "target",
                    "task_type": "binary_classification",
                },
            )

        # Client 2: low-level mcp.client.stdio.stdio_client + ClientSession
        async with (
            stdio_client(_stdio_params(session_file)) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as client2,
        ):
            await client2.initialize()
            listed2 = await client2.list_tools()
            c2_tools = [t.name for t in listed2.tools]
            c2_inspect = await client2.call_tool("inspect_dataset", {})
            c2_metrics = await client2.call_tool("get_experiment_metrics", {"job_id": job_id})

        def _actionable(res: Any) -> bool:
            sc = res.structured_content or {}
            return bool(res.is_error) and bool(
                sc.get("code") and sc.get("message") and sc.get("suggested_recovery")
            )

        checks = {
            "session_file_mode_0600": file_mode == 0o600,
            "secret_not_in_session_file": secret_hidden,
            "missing_secret_rejected": proc_missing.returncode != 0,
            "wrong_secret_rejected": proc_wrong.returncode != 0,
            "client1_mcp_Client_10_tools": len(c1_tools) == 10 and not c1_inspect.is_error,
            "client2_ClientSession_10_tools": (
                len(c2_tools) == 10 and not c2_inspect.is_error and not c2_metrics.is_error
            ),
            "actionable_error_unknown_args": _actionable(err_unknown),
            "actionable_error_idempotency": _actionable(err_idem),
            "actionable_error_missing_trial": _actionable(err_missing),
            "actionable_error_diag_missing_target": _actionable(err_diag),
            "diag_recovered_after_error": not recovered_diag.is_error,
        }
        return {"all_passed": all(checks.values()), "checks": checks}


def _check_task_args(
    task: dict[str, Any],
    args: dict[str, Any],
    *,
    seeded_trial_id: str,
    seeded_job_id: str,
) -> bool:
    checks = task.get("arg_checks") or {}
    if "dataset_id" in checks and args.get("dataset_id") != checks["dataset_id"]:
        return False
    if "diagnostic" in checks and args.get("diagnostic") != checks["diagnostic"]:
        return False
    if "column" in checks and args.get("column") != checks["column"]:
        return False
    if checks.get("nonempty_query") and not str(args.get("query") or "").strip():
        return False
    if checks.get("has_pipeline_py") and "def fit" not in str(args.get("pipeline_py") or ""):
        return False
    if checks.get("trial_id_key") == "seeded_trial_id" and args.get("trial_id") != seeded_trial_id:
        return False
    if checks.get("job_id_key") == "seeded_job_id" and args.get("job_id") != seeded_job_id:
        return False
    return not ("tail_lines" in checks and args.get("tail_lines") != checks["tail_lines"])


async def _complete_resilient(
    adapter: LLMAdapter,
    messages: Sequence[Message],
    *,
    context: CallContext,
    tools: Sequence[ToolSpec],
) -> Any:
    """Call `adapter.complete` with one outer 30s backoff on transient 503 spikes."""
    try:
        return await adapter.complete(messages, tier=ModelTier.LEAD, context=context, tools=tools)
    except LLMCallError as exc:
        if "503" not in str(exc) and "UNAVAILABLE" not in str(exc):
            raise
        print(f"[F5] Transient 503 on {context.prompt_ref}, sleeping 30s ...", flush=True)
        await asyncio.sleep(30.0)
        return await adapter.complete(messages, tier=ModelTier.LEAD, context=context, tools=tools)


async def _run_model_arm(
    *,
    model_label: str,
    profile: ModelProfile,
    desc_version: str,
    tasks: Sequence[dict[str, Any]],
    repeat_tasks: Sequence[dict[str, Any]],
    adapter: LLMAdapter,
    cap_usd: float | None,
) -> dict[str, Any]:
    safe_label = model_label.replace("/", "_")
    with tempfile.TemporaryDirectory(prefix=f"f5_{safe_label}_{desc_version}_") as tmp:
        session_file, run_id, seeded_trial_id, seeded_job_id = _seed_workspace(Path(tmp) / "ws")
        paid_usd = 0.0
        list_usd = 0.0
        in_tok = 0
        out_tok = 0

        async with Client(_stdio_params(session_file), read_timeout_seconds=60) as client:
            listed = await client.list_tools()
            prod_specs = [
                ToolSpec.from_mcp(t.model_dump(by_alias=True))
                for t in listed.tools
                if t.name in OFFLINE_TOOLS
            ]
            active_specs = (
                prod_specs
                if desc_version == "v1_production"
                else strip_tool_descriptions(prod_specs)
            )

            async def run_one(task: dict[str, Any], tag: str) -> dict[str, Any]:
                nonlocal paid_usd, list_usd, in_tok, out_tok
                if cap_usd is not None and paid_usd >= cap_usd:
                    raise RuntimeError(f"OpenAI spend cap ${cap_usd:.2f} reached (${paid_usd:.4f})")

                t0 = time.perf_counter()
                prompt = str(task["prompt_template"]).format(
                    run_id=run_id, trial_id=seeded_trial_id, job_id=seeded_job_id
                )
                sys_msg = Message.system(
                    "You are a Hypora ML assistant. Call exactly the one appropriate MCP tool "
                    "for the user's request with valid arguments."
                )
                usr_msg = Message.user(prompt)
                messages: list[Message] = [sys_msg, usr_msg]
                ctx = CallContext(
                    run_id=f"f5_{model_label}_{desc_version}",
                    actor_role=ActorRole.SINGLE_AGENT,
                    prompt_ref=f"{task['id']}_{tag}",
                )
                resp1 = await _complete_resilient(
                    adapter, messages, context=ctx, tools=active_specs
                )
                call_cost = float(resp1.total_cost_usd)
                list_usd += call_cost
                if profile is ModelProfile.EVAL:
                    paid_usd += call_cost
                in_tok += int(resp1.usage.input_tokens)
                out_tok += int(resp1.usage.output_tokens)

                first_try_ok = False
                final_ok = False
                called_tool: str | None = None
                err_code: str | None = None
                recovery_err_code: str | None = None
                recovery_attempted = False

                if resp1.tool_calls:
                    call1 = resp1.tool_calls[0]
                    called_tool = call1.name
                    res1 = await client.call_tool(call1.name, dict(call1.arguments))
                    args_ok1 = _check_task_args(
                        task,
                        dict(call1.arguments),
                        seeded_trial_id=seeded_trial_id,
                        seeded_job_id=seeded_job_id,
                    )
                    first_try_ok = (
                        call1.name == task["expected_tool"] and not res1.is_error and args_ok1
                    )
                    if res1.is_error:
                        err_code = str((res1.structured_content or {}).get("code") or "ERROR")
                    if first_try_ok:
                        final_ok = True
                    else:
                        recovery_attempted = True
                        if cap_usd is not None and paid_usd >= cap_usd:
                            raise RuntimeError(
                                f"OpenAI spend cap ${cap_usd:.2f} reached (${paid_usd:.4f})"
                            )
                        obs_payload = res1.structured_content or {
                            "error": "Tool call did not match task requirements."
                        }
                        if not res1.is_error and not args_ok1:
                            obs_payload = {
                                "error": "Arguments did not satisfy the task constraints.",
                                "received_tool": call1.name,
                            }
                        messages.append(resp1.message)
                        messages.append(
                            Message.tool_result(
                                call1,
                                json.dumps(obs_payload),
                                is_error=res1.is_error or not args_ok1,
                            )
                        )
                        resp2 = await _complete_resilient(
                            adapter,
                            messages,
                            context=CallContext(
                                run_id=f"f5_{model_label}_{desc_version}",
                                actor_role=ActorRole.SINGLE_AGENT,
                                prompt_ref=f"{task['id']}_{tag}_rec",
                            ),
                            tools=active_specs,
                        )
                        c2_cost = float(resp2.total_cost_usd)
                        list_usd += c2_cost
                        if profile is ModelProfile.EVAL:
                            paid_usd += c2_cost
                        in_tok += int(resp2.usage.input_tokens)
                        out_tok += int(resp2.usage.output_tokens)
                        call_cost += c2_cost
                        if resp2.tool_calls:
                            call2 = resp2.tool_calls[0]
                            res2 = await client.call_tool(call2.name, dict(call2.arguments))
                            args_ok2 = _check_task_args(
                                task,
                                dict(call2.arguments),
                                seeded_trial_id=seeded_trial_id,
                                seeded_job_id=seeded_job_id,
                            )
                            final_ok = (
                                call2.name == task["expected_tool"]
                                and not res2.is_error
                                and args_ok2
                            )
                            if res2.is_error:
                                recovery_err_code = str(
                                    (res2.structured_content or {}).get("code") or "ERROR"
                                )

                elapsed_s = time.perf_counter() - t0
                return {
                    "task_id": task["id"],
                    "expected_tool": task["expected_tool"],
                    "called_tool": called_tool,
                    "tag": tag,
                    "first_try_ok": first_try_ok,
                    "final_ok": final_ok,
                    "recovered": (not first_try_ok) and final_ok,
                    "recovery_attempted": recovery_attempted,
                    "error_code": err_code,
                    "recovery_error_code": recovery_err_code,
                    "latency_s": round(elapsed_s, 3),
                    "list_cost_usd": round(call_cost, 6),
                    "paid_cost_usd": round(call_cost if profile is ModelProfile.EVAL else 0.0, 6),
                }

            primary = [await run_one(task, "primary") for task in tasks]
            repeats: list[dict[str, Any]] = []
            for rep in range(1, 4):
                for task in repeat_tasks:
                    repeats.append(await run_one(task, f"repeat_{rep}"))

        return {
            "model": model_label,
            "profile": profile.value,
            "description_version": desc_version,
            "primary": primary,
            "repeats": repeats,
            "spend": {
                "paid_usd": round(paid_usd, 6),
                "list_price_usd": round(list_usd, 6),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            },
        }


def _summarize_arm(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = [bool(r["first_try_ok"]) for r in records]
    final = [bool(r["final_ok"]) for r in records]
    w1 = wilson_interval(sum(first), len(first))
    w2 = wilson_interval(sum(final), len(final))
    lat = latency_summary([float(r["latency_s"]) for r in records])
    cost = cost_per_task([float(r["list_cost_usd"]) for r in records])
    return {
        "n": len(records),
        "first_try_ok": sum(first),
        "first_try_rate": round(sum(first) / len(first), 6),
        "first_try_wilson_95": [round(w1.low, 6), round(w1.high, 6)],
        "final_ok": sum(final),
        "final_rate": round(sum(final) / len(final), 6),
        "final_wilson_95": [round(w2.low, 6), round(w2.high, 6)],
        "recovered_count": sum(1 for r in records if r["recovered"]),
        "recovery_attempted": sum(1 for r in records if r["recovery_attempted"]),
        "latency_s": {"mean": round(lat.mean, 3), "p95": round(lat.p95, 3), "method": lat.method},
        "list_cost_per_task_usd": round(cost.per_task_usd, 6),
    }


def _pair_comp(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    diff = paired_bootstrap_difference(a, b, seed=1990)
    tbl = paired_outcome_table(a, b)
    return {
        "difference": round(diff.difference, 6),
        "ci_low": round(diff.interval.low, 6),
        "ci_high": round(diff.interval.high, 6),
        "verdict": diff.verdict.value,
        "resamples": diff.resamples,
        "seed": diff.seed,
        "outcome_table": {
            "both_pass": tbl.both_pass,
            "first_only": tbl.first_only,
            "second_only": tbl.second_only,
            "both_fail": tbl.both_fail,
        },
    }


def _build_summary_md(ev: dict[str, Any]) -> str:
    arms = ev["arms"]
    comps = ev["comparisons"]
    sp = ev["spend"]
    lines = [
        "# F5 MCP Server Evaluation Summary (S5.3)",
        "",
        f"- **Benchmark:** {ev['total_tasks']} typed MCP tasks across 8 real stdio tools "
        "(`python -m hypora.mcp_server`), plus 6x3 separate repeats",
        "- **Models:** `openai/gpt-6-luna` (`eval` profile, paid) and "
        "`gemini/gemini-3.5-flash-lite` (`dev` profile, free tier)",
        "- **Clients & Auth:** Dual stdio client verification (`mcp.Client` + low-level "
        "`ClientSession`), mode-`0600` hash-only secret file, missing/wrong secret refusal, and "
        "actionable `ToolError` checks",
        f"- **Spend:** Paid OpenAI `${sp['total_paid_usd']:.6f}` (`cap=${CAP_USD:.2f}`); "
        f"combined list-price estimate `${sp['total_list_price_usd']:.6f}` "
        f"(price date `{PRICE_RETRIEVED_ON}`)",
        "",
        "## 1. Primary Task Success (24 Tasks x 2 Models x 2 Description Versions)",
        "",
        "| Model | Descriptions | First-Try OK (Wilson 95% CI) | "
        "Post-Recovery OK (Wilson 95% CI) | Recovered | Mean / p95 Latency (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in (
        "openai_v0_minimal",
        "openai_v1_production",
        "gemini_v0_minimal",
        "gemini_v1_production",
    ):
        s = arms[key]["primary_summary"]
        w1_lo, w1_hi = s["first_try_wilson_95"][0] * 100, s["first_try_wilson_95"][1] * 100
        w2_lo, w2_hi = s["final_wilson_95"][0] * 100, s["final_wilson_95"][1] * 100
        lines.append(
            f"| `{arms[key]['model']}` | `{arms[key]['description_version']}` | "
            f"{s['first_try_ok']}/24 ({s['first_try_rate'] * 100:.1f}%, "
            f"[{w1_lo:.1f}%, {w1_hi:.1f}%]) | "
            f"{s['final_ok']}/24 ({s['final_rate'] * 100:.1f}%, "
            f"[{w2_lo:.1f}%, {w2_hi:.1f}%]) | "
            f"{s['recovered_count']}/{s['recovery_attempted']} | "
            f"{s['latency_s']['mean']:.2f} / {s['latency_s']['p95']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Paired Bootstrap Comparisons (10,000 resamples, `seed=1990`)",
            "",
            "| Comparison | Delta | 95% Bootstrap CI | Both / First Only / Second Only / Neither |",
            "|---|---:|---:|---|",
        ]
    )
    for cname, c in comps.items():
        t = c["outcome_table"]
        lines.append(
            f"| `{cname}` | {c['difference']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | "
            f"{t['both_pass']} / {t['first_only']} / {t['second_only']} / {t['both_fail']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Repeat Sample (6x3 = 18 Trials per Arm)",
            "",
            "| Arm | First-Try OK | Post-Recovery OK | Wilson 95% CI (Final) |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in (
        "openai_v0_minimal",
        "openai_v1_production",
        "gemini_v0_minimal",
        "gemini_v1_production",
    ):
        rs = arms[key]["repeat_summary"]
        lines.append(
            f"| `{key}` | {rs['first_try_ok']}/18 | "
            f"{rs['final_ok']}/18 ({rs['final_rate'] * 100:.1f}%) | "
            f"[{rs['final_wilson_95'][0] * 100:.1f}%, {rs['final_wilson_95'][1] * 100:.1f}%] |"
        )

    auth = ev["auth_and_dual_client"]
    lines.extend(
        [
            "",
            "## 4. Dual-Client, Session Authentication & Actionable Validation Checks",
            "",
            f"- All 11 deterministic stdio checks passed: **{auth['all_passed']}**",
            f"  (`{json.dumps(auth['checks'])}`)",
            "",
            "## 5. Honest Limitations",
            "",
            "- Offline-capable MCP tools only (8 of 10 tools): `search_model_catalog` and "
            "`get_model_evidence` query external HuggingFace endpoints and are excluded from the "
            "live 24-task runner to keep evaluation reproducible and offline-safe (still listed in "
            "both stdio clients' 10-tool inventories).",
            "- Gemini `dev` profile runs on Google's free tier (`$0` actual spend) at client-paced "
            "RPM headroom, so Gemini latency reflects rate-limit pacing plus inference time.",
        ]
    )
    return "\n".join(lines) + "\n"


async def run_full_evaluation() -> dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env", override=False)
    fallback_env = Path("/home/ghadi/hypora/.env")
    if fallback_env.exists():
        load_dotenv(fallback_env, override=False)

    benchmark = load_tasks()
    tasks = list(benchmark["tasks"])
    repeat_ids = set(benchmark["repeat_task_ids"])
    repeat_tasks = [t for t in tasks if t["id"] in repeat_ids]

    print("[F5] Running deterministic stdio dual-client, auth, and error checks ...", flush=True)
    auth_checks = await verify_stdio_auth_and_clients()
    if not auth_checks["all_passed"]:
        raise RuntimeError(f"Stdio auth/client checks failed: {auth_checks}")

    base_settings = load_settings()
    arms: dict[str, Any] = {}

    ckpt_dir = Path(tempfile.gettempdir()) / "f5_eval_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for model_key, profile, model_label, cap in (
        ("openai", ModelProfile.EVAL, "openai/gpt-6-luna", CAP_USD),
        ("gemini", ModelProfile.DEV, "gemini/gemini-3.5-flash-lite", None),
    ):
        settings = base_settings.model_copy(update={"model_profile": profile})
        run_config = RunConfig.build(settings, hardware=HARDWARE)
        sink = _MemorySink()
        adapter = LLMAdapter.from_run_config(
            run_config, events=sink, admission=ProviderAdmission(run_config.models.rate_limits)
        )
        for desc_ver in ("v0_minimal", "v1_production"):
            arm_key = f"{model_key}_{desc_ver}"
            ckpt_file = ckpt_dir / f"{arm_key}.json"
            if ckpt_file.exists():
                print(f"[F5] Reusing cached checkpoint {arm_key} ...", flush=True)
                arms[arm_key] = json.loads(ckpt_file.read_text(encoding="utf-8"))
                continue
            print(f"[F5] Running {model_label} ({desc_ver}) ...", flush=True)
            res = await _run_model_arm(
                model_label=model_label,
                profile=profile,
                desc_version=desc_ver,
                tasks=tasks,
                repeat_tasks=repeat_tasks,
                adapter=adapter,
                cap_usd=cap,
            )
            res["primary_summary"] = _summarize_arm(res["primary"])
            res["repeat_summary"] = _summarize_arm(res["repeats"])
            arms[arm_key] = res
            ckpt_file.write_text(json.dumps(res, indent=2), encoding="utf-8")

    comparisons = {
        "openai_v1_vs_v0_first_try": _pair_comp(
            [bool(r["first_try_ok"]) for r in arms["openai_v1_production"]["primary"]],
            [bool(r["first_try_ok"]) for r in arms["openai_v0_minimal"]["primary"]],
        ),
        "openai_v1_vs_v0_final": _pair_comp(
            [bool(r["final_ok"]) for r in arms["openai_v1_production"]["primary"]],
            [bool(r["final_ok"]) for r in arms["openai_v0_minimal"]["primary"]],
        ),
        "gemini_v1_vs_v0_first_try": _pair_comp(
            [bool(r["first_try_ok"]) for r in arms["gemini_v1_production"]["primary"]],
            [bool(r["first_try_ok"]) for r in arms["gemini_v0_minimal"]["primary"]],
        ),
        "gemini_v1_vs_v0_final": _pair_comp(
            [bool(r["final_ok"]) for r in arms["gemini_v1_production"]["primary"]],
            [bool(r["final_ok"]) for r in arms["gemini_v0_minimal"]["primary"]],
        ),
        "openai_vs_gemini_v1_first_try": _pair_comp(
            [bool(r["first_try_ok"]) for r in arms["openai_v1_production"]["primary"]],
            [bool(r["first_try_ok"]) for r in arms["gemini_v1_production"]["primary"]],
        ),
        "openai_vs_gemini_v1_final": _pair_comp(
            [bool(r["final_ok"]) for r in arms["openai_v1_production"]["primary"]],
            [bool(r["final_ok"]) for r in arms["gemini_v1_production"]["primary"]],
        ),
    }

    total_paid = sum(float(a["spend"]["paid_usd"]) for a in arms.values())
    total_list = sum(float(a["spend"]["list_price_usd"]) for a in arms.values())
    evidence: dict[str, Any] = {
        "feature": "F5",
        "benchmark": benchmark["benchmark"],
        "total_tasks": len(tasks),
        "repeat_task_ids": list(benchmark["repeat_task_ids"]),
        "price_retrieved_on": PRICE_RETRIEVED_ON,
        "auth_and_dual_client": auth_checks,
        "arms": arms,
        "comparisons": comparisons,
        "spend": {
            "openai_cap_usd": CAP_USD,
            "total_paid_usd": round(total_paid, 6),
            "total_list_price_usd": round(total_list, 6),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "f5-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "f5-summary.md").write_text(_build_summary_md(evidence), encoding="utf-8")
    print(
        f"[F5] Complete! Paid OpenAI=${total_paid:.6f}, List-price=${total_list:.6f}",
        flush=True,
    )
    return evidence


if __name__ == "__main__":
    asyncio.run(run_full_evaluation())
