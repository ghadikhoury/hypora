"""S4.2 recorded readiness: the six dev cases on the pinned eval profile, multi and single (D-051).

The same real controller, MCP server, worker, sandbox, hidden splits and programmatic graders as
`make eval`, through the same runner. Spending reuses the S2.6 collector's durable ledger
(`scripts/stage2_eval.py`): every actual OpenAI SDK attempt, retries included, reserves $0.36
before dispatch and settles only provider-reported usage, so unknown charges keep their full
reservation across crashes. This collection has its own journal and the owner's $10 aggregate
cap. Each case run keeps its case budget ($0.60), each comparison refuses to start when its runs
could pass $5 at full budgets, and no comparison starts when its full budgets plus everything
already spent or reserved could pass the $10 cap. No prompts, rows or credentials enter the audit.
"""

import argparse
import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import anyio

from hypora.cli.eval import EvalOptions, build_plan
from hypora.core.config import load_settings_for_eval
from hypora.core.hashing import sha256_hex
from hypora.core.roles.prompt_loader import PROMPTS_DIR, load_prompt
from hypora.core.schemas import ModelProfile
from hypora.evaluation.runner.agent import AGENT_SYSTEM
from hypora.evaluation.runner.runner import run_eval
from hypora.evaluation.runner.store import EvalRefused
from scripts.stage2_eval import (
    AccountedAgent,
    AdapterFactory,
    SpendError,
    SpendLedger,
    check_published_history,
    recorded_code,
)

ROOT: Final = Path(__file__).resolve().parents[1]
STATE: Final = ROOT / "runs" / "readiness"
LEDGER: Final = STATE / "paid-attempts.jsonl"
AUDIT: Final = ROOT / "results" / "readiness" / "paid-accounting.json"
LOCK: Final = ROOT / "results" / "readiness" / "lock.json"
TOOLS_CONTRACT: Final = ROOT / "docs" / "mcp" / "tools.json"
CAP_USD: Final = 10.0
"""The owner's aggregate cap for the whole readiness collection, retries included (D-051)."""
RUN_CAP_USD: Final = 5.0
"""At most this at full case budgets in one comparison (PLAN.md §0.3 trigger 6)."""
CONFIGS: Final = ("default", "single_agent")
"""`default` is the multi-agent A/B/C system; `single_agent` its F3 comparison point."""
SYSTEMS: Final = ("agent", "baseline")


def current_lock() -> dict[str, Any]:
    """What S4.2 locks (D-051): the eval models and retry policy, both configs' flags and limits,
    the default budgets, every runtime prompt's version and SHA-256, and the MCP tool contract.

    Hardware-derived resource pools are left out, so the lock reads the same on every machine.
    """
    configs: dict[str, Any] = {}
    for config in CONFIGS:
        settings = load_settings_for_eval(config)
        configs[config] = {
            "model_profile": settings.model_profile.value,
            "flags": settings.flags.model_dump(mode="json"),
        }
    settings = load_settings_for_eval(CONFIGS[0])
    prompts = {
        path.stem: {
            "version": load_prompt(path.stem).version,
            "sha256": sha256_hex(path.read_bytes()),
        }
        for path in sorted(PROMPTS_DIR.glob("*.md"))
    }
    return {
        "step": "S4.2",
        "decision": "D-051",
        "eval_profile": settings.models.eval.model_dump(mode="json"),
        "retry": settings.models.retry.model_dump(mode="json"),
        "configs": configs,
        "budgets": settings.budgets.model_dump(mode="json"),
        "prompts": prompts,
        "tools_contract_sha256": sha256_hex(TOOLS_CONTRACT.read_bytes()),
    }


def write_lock() -> Path:
    """Record the lock beside the readiness results."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(current_lock(), indent=2, sort_keys=True) + "\n")
    return LOCK


def ledger() -> SpendLedger:
    """The readiness journal under ignored runs/, published to results/readiness/."""
    return SpendLedger(LEDGER, cap_usd=CAP_USD, audit=AUDIT)


def check_aggregate(paid: SpendLedger, worst_usd: float) -> None:
    """Refuse a comparison whose full case budgets could carry the collection past its cap."""
    state = paid.snapshot()
    exposure = state.measured_usd + state.reserved_usd
    if exposure + worst_usd > paid.cap_usd:
        raise SpendError(
            f"${exposure:.4f} spent or reserved plus ${worst_usd:.2f} at full case budgets "
            f"would pass the ${paid.cap_usd:.2f} cap; no comparison was started"
        )


async def comparison(config: str, systems: tuple[str, ...], resume: bool, paid: SpendLedger) -> int:
    """Six actual dev cases on the chosen systems with the existing comparison runner."""
    recorded_code()
    plan = build_plan(
        EvalOptions(
            cases="dev",
            config=config,
            systems=systems,
            profile=ModelProfile.EVAL,
            parallel=1,
            resume=resume,
            max_total_usd=RUN_CAP_USD,
        ),
        print,
    )
    if AGENT_SYSTEM in systems:
        check_aggregate(paid, sum(case.budget.max_usd or 0.0 for case in plan.cases))
        factory = AdapterFactory(paid)
        plan = replace(
            plan,
            systems=tuple(
                AccountedAgent(factory) if system.name == AGENT_SYSTEM else system
                for system in plan.systems
            ),
        )
    outcome = await run_eval(plan)
    return 0 if outcome.clean else 1


@contextmanager
def collection_lock() -> Iterator[None]:
    """Only one readiness collector publishes records at once."""
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "collection.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SpendError("another readiness collector is already running") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    """The owner-approved S4.2 entry point; there is no cap increase or model remap."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", choices=CONFIGS, required=True)
    run.add_argument("--systems", choices=("agent,baseline", "baseline"), default="agent,baseline")
    run.add_argument("--resume", action="store_true")
    commands.add_parser("spend")
    commands.add_parser("lock")
    args = parser.parse_args(argv)
    if args.command == "lock":
        print(write_lock())
        return 0
    paid = ledger()
    try:
        if args.command == "spend":
            print(paid.publish())
            return 0
        with collection_lock():
            check_published_history(paid)
            systems = tuple(args.systems.split(","))
            return anyio.run(comparison, args.config, systems, args.resume, paid)
    except (SpendError, EvalRefused) as exc:
        print(f"Collection refused: {exc}")
        return 2
    finally:
        try:
            paid.publish()
        except SpendError:
            print("Paid ledger needs inspection; no reservation released; audit not regenerated.")


if __name__ == "__main__":
    raise SystemExit(main())
