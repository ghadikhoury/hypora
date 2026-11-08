"""S2.6 recorded dev/eval comparisons and matching pinned-eval S1.7 smokes.

Uses the existing real controller, worker, sandbox and graders. Every paid SDK attempt reserves
$0.36 in one durable ledger BEFORE dispatch, including retries and both smoke tasks. Unknown
charges retain their reservation across crashes/resumes. No prompts or credentials enter it.
"""

import argparse
import fcntl
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import anyio

from hypora.cli.eval import EvalOptions, build_plan
from hypora.cli.run import AdapterFactory as CLIAdapterFactory
from hypora.cli.run import RunOptions, execute
from hypora.core.config import load_settings_for_eval
from hypora.core.config.run_config import RunConfig
from hypora.core.llm import LLMAdapter, ProviderAdmission, load_price_table
from hypora.core.llm.adapter import build_client
from hypora.core.llm.credentials import load_credentials
from hypora.core.llm.errors import ProviderError, ProviderErrorKind
from hypora.core.llm.providers.base import ProviderClient, ProviderRequest, ProviderResult
from hypora.core.llm.sinks import FileModelIOStore
from hypora.core.schemas import ApprovalPolicy, ModelProfile, ModelTier, Provider
from hypora.evaluation.runner.agent import AgentSystem
from hypora.evaluation.runner.runner import run_eval
from hypora.evaluation.runner.store import EvalRefused, GitRepo
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.snapshots import SNAPSHOTS_DIR
from scripts import stage1_smoke as smoke

ROOT: Final = Path(__file__).resolve().parents[1]
STATE: Final = ROOT / "runs" / "stage2"
LEDGER: Final = STATE / "paid-attempts.jsonl"
CAP_USD: Final = 8.0
ATTEMPT_RESERVATION_USD: Final = 0.36
"""Luna absolute bounds: 1.05M input x $0.25/M + 128K output x $0.75/M = $0.3585.

The input bound uses the highest published long-context input/cache-write rate. This deliberately
reserves the whole model limit instead of estimating token counts from characters or encrypted
reasoning. Model requests, output limits, dev behavior and adapter retry policy are unchanged.
"""
SMOKE_RUN_BUDGET_USD: Final = 0.45


class SpendError(Exception):
    """The paid ledger cannot safely admit another attempt."""


@dataclass(frozen=True)
class SpendSnapshot:
    """Measured charges and conservative exposure, never conflated with one another."""

    measured_usd: float
    reserved_usd: float
    attempts: tuple[dict[str, Any], ...]


class SpendLedger:
    """Append-only, cross-process locked, fsync'ed reservations; malformed history fails closed."""

    def __init__(self, path: Path = LEDGER, cap_usd: float = CAP_USD) -> None:
        if not math.isfinite(cap_usd) or cap_usd <= 0:
            raise SpendError("invalid aggregate cap")
        self.path, self.cap_usd = path, cap_usd

    @contextmanager
    def _locked(self) -> Iterator[Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield stream
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @staticmethod
    def _append(stream: Any, event: dict[str, Any]) -> None:
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        folder = Path(stream.name).parent
        for directory in (folder, *folder.parents):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _read(self, stream: Any) -> dict[str, dict[str, Any]]:
        stream.seek(0)
        content = stream.read()
        if not content:
            self._append(stream, {"kind": "cap", "version": 1, "cap_usd": self.cap_usd})
            return {}
        if not content.endswith("\n"):
            raise SpendError("truncated paid ledger; inspect accounting before any new paid call")
        try:
            events = [json.loads(line) for line in content.splitlines()]
            if events[0] != {"kind": "cap", "version": 1, "cap_usd": self.cap_usd}:
                raise ValueError("cap/header changed")
            attempts: dict[str, dict[str, Any]] = {}
            for event in events[1:]:
                if event["kind"] == "reserve":
                    if set(event) != {"kind", "id", "run_id", "maximum_usd", "at"}:
                        raise ValueError("invalid reservation fields")
                    if (
                        type(event["maximum_usd"]) not in (int, float)
                        or not isinstance(event["id"], str)
                        or not isinstance(event["run_id"], str)
                    ):
                        raise ValueError("invalid reservation types")
                    maximum = float(event["maximum_usd"])
                    if (
                        not math.isfinite(maximum)
                        or maximum != ATTEMPT_RESERVATION_USD
                        or event["id"] in attempts
                    ):
                        raise ValueError("invalid/duplicate reservation")
                    attempts[event["id"]] = {**event, "actual_usd": None}
                elif event["kind"] == "settle":
                    if set(event) != {"kind", "id", "actual_usd", "at", "usage", "price"}:
                        raise ValueError("invalid settlement fields")
                    row = attempts[event["id"]]
                    if type(event["actual_usd"]) not in (int, float):
                        raise ValueError("invalid settlement type")
                    actual = float(event["actual_usd"])
                    if not math.isfinite(actual) or actual < 0 or row["actual_usd"] is not None:
                        raise ValueError("invalid/duplicate settlement")
                    if not isinstance(event["usage"], dict) or not isinstance(event["price"], dict):
                        raise ValueError("missing cost evidence")
                    self._validate_settlement(actual, event["usage"], event["price"])
                    row["actual_usd"] = actual
                    row["usage"] = event["usage"]
                    row["price"] = event["price"]
                else:
                    raise ValueError("unknown event")
            return attempts
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise SpendError(
                "malformed paid ledger; inspect accounting before new paid calls"
            ) from exc

    @staticmethod
    def _validate_settlement(actual: float, usage: dict[str, Any], price: dict[str, Any]) -> None:
        # Pydantic's Usage defaults and coercions are useful elsewhere but unsafe for a
        # persisted bill: missing or coercible counters cannot establish known spend.
        from hypora.core.llm.pricing import ModelPrice
        from hypora.core.llm.types import Usage

        fields = {
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        }
        if set(usage) != fields or any(type(usage[k]) is not int or usage[k] < 0 for k in fields):
            raise ValueError("invalid or incomplete usage counters")
        computed = ModelPrice.model_validate(price).cost_usd(Usage.model_validate(usage))
        if not math.isclose(computed, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("charge differs from usage/price evidence")

    @staticmethod
    def _snapshot(attempts: dict[str, dict[str, Any]]) -> SpendSnapshot:
        return SpendSnapshot(
            measured_usd=sum(row["actual_usd"] or 0.0 for row in attempts.values()),
            reserved_usd=sum(
                row["maximum_usd"] for row in attempts.values() if row["actual_usd"] is None
            ),
            attempts=tuple(attempts.values()),
        )

    def snapshot(self) -> SpendSnapshot:
        """Read durable known/unknown exposure; a crash never releases a reservation."""
        with self._locked() as stream:
            return self._snapshot(self._read(stream))

    def reserve(self, run_id: str) -> str:
        """Reserve the absolute worst-case price before one actual paid provider attempt."""
        with self._locked() as stream:
            state = self._snapshot(self._read(stream))
            if any(
                row["actual_usd"] is not None and row["actual_usd"] > row["maximum_usd"]
                for row in state.attempts
            ):
                raise SpendError("a prior charge exceeded its bound; inspect accounting")
            if state.measured_usd + state.reserved_usd + ATTEMPT_RESERVATION_USD > self.cap_usd:
                raise SpendError("aggregate paid cap reached; no provider call was dispatched")
            identifier = uuid4().hex
            self._append(
                stream,
                {
                    "kind": "reserve",
                    "id": identifier,
                    "run_id": run_id,
                    "maximum_usd": ATTEMPT_RESERVATION_USD,
                    "at": datetime.now(UTC).isoformat(),
                },
            )
            return identifier

    def settle(
        self,
        identifier: str,
        actual_usd: float,
        *,
        usage: dict[str, Any] | None = None,
        price: dict[str, Any] | None = None,
    ) -> None:
        """Settle only provider-reported usage. Missing/error/cancelled usage remains reserved."""
        if not math.isfinite(actual_usd) or actual_usd < 0:
            raise SpendError("invalid reported charge")
        if usage is None or price is None:
            raise SpendError("settlement requires provider usage and price evidence")
        try:
            self._validate_settlement(actual_usd, usage, price)
        except (TypeError, ValueError, KeyError) as exc:
            raise SpendError("invalid settlement evidence") from exc
        with self._locked() as stream:
            attempts = self._read(stream)
            if identifier not in attempts:
                raise SpendError("settlement has no reservation")
            row = attempts[identifier]
            if row["actual_usd"] is not None:
                if row["actual_usd"] == actual_usd:
                    return
                raise SpendError("conflicting settlement")
            self._append(
                stream,
                {
                    "kind": "settle",
                    "id": identifier,
                    "actual_usd": actual_usd,
                    "usage": usage,
                    "price": price,
                    "at": datetime.now(UTC).isoformat(),
                },
            )
            if actual_usd > row["maximum_usd"]:
                raise SpendError(
                    "reported charge exceeded verified model price bound; stop collection"
                )

    def unobserved(self, run_id: str) -> float:
        """Conservative unknown charges for one run, as displayed in its result record."""
        return sum(
            row["maximum_usd"]
            for row in self.snapshot().attempts
            if row["run_id"] == run_id and row["actual_usd"] is None
        )

    def publish(self) -> Path:
        """Save a public accounting audit containing IDs/costs only, never model input/output."""
        check_published_history(self)
        state = self.snapshot()
        path = ROOT / "results" / "stage2" / "paid-accounting.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cap_usd": self.cap_usd,
            "attempt_reservation_usd": ATTEMPT_RESERVATION_USD,
            "measured_usd": state.measured_usd,
            "unknown_reserved_usd": state.reserved_usd,
            "attempts": state.attempts,
        }
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        return path


class ReservedClient:
    """One reserve per real SDK attempt, so retries are protected without changing their policy."""

    def __init__(self, inner: ProviderClient, ledger: SpendLedger, run_id: str) -> None:
        if inner.provider is not Provider.OPENAI:
            raise SpendError("paid wrapper requires the OpenAI SDK client")
        self.inner, self.ledger, self.run_id = inner, ledger, run_id
        self.price = load_price_table().require(Provider.OPENAI, "gpt-6-luna")
        prices = [self.price.standard, self.price.long_context]
        if any(
            p is not None
            and (
                max(
                    p.input_usd_per_mtok or 0,
                    p.cached_input_usd_per_mtok or 0,
                    p.cache_write_usd_per_mtok or 0,
                )
                > 0.25
                or (p.output_usd_per_mtok or 0) > 0.75
            )
            for p in prices
        ):
            raise SpendError("prices exceed the verified Luna reservation bound")

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        """Dispatch only after durable admission; ambiguous failures keep their full exposure."""
        if request.model_id != "gpt-6-luna":
            raise SpendError("the paid reservation bound applies to gpt-6-luna only")
        try:
            identifier = self.ledger.reserve(self.run_id)
        except SpendError as exc:
            raise ProviderError(ProviderErrorKind.QUOTA, str(exc)) from None
        try:
            result = await self.inner.generate(request)
        except ProviderError as exc:
            if exc.usage is not None:
                self.ledger.settle(
                    identifier,
                    self.price.cost_usd(exc.usage),
                    usage=exc.usage.model_dump(mode="json"),
                    price=self.price.model_dump(mode="json"),
                )
            raise
        # Cancellation/unknown exceptions deliberately leave the reservation in place.
        self.ledger.settle(
            identifier,
            self.price.cost_usd(result.usage),
            usage=result.usage.model_dump(mode="json"),
            price=self.price.model_dump(mode="json"),
        )
        return result


def build_paid_client(key: str, timeout: float) -> ProviderClient:
    """Pin S2.6 to the global Standard endpoint used in the spend bound."""
    from openai import AsyncOpenAI

    from hypora.core.llm.providers.openai_client import OpenAIClient

    sdk = AsyncOpenAI(
        api_key=key,
        base_url="https://api.openai.com/v1",
        max_retries=0,
        timeout=timeout,
    )
    return OpenAIClient(key, timeout_s=timeout, client=sdk, service_tier="default")


class AdapterFactory:
    """Real clients and shared admission, with the paid ledger applied to every OpenAI attempt."""

    def __init__(self, ledger: SpendLedger) -> None:
        self.ledger = ledger
        self.admission: ProviderAdmission | None = None

    def for_run(
        self, config: RunConfig, database: Database, runs_root: Path, run_id: str
    ) -> LLMAdapter:
        """Build the standard adapter; dev requests and model parameters are unchanged."""
        if config.active_profile.fallback.enabled:
            raise SpendError("S2.6 collection requires fallback disabled on both profiles")
        credentials = load_credentials()
        providers = {config.active_profile.tiers.for_tier(tier).provider for tier in ModelTier}
        clients: dict[Provider, ProviderClient] = {}
        for provider in providers:
            if provider is Provider.OPENAI:
                client = build_paid_client(
                    credentials.api_key(provider), config.models.retry.request_timeout_s
                )
                clients[provider] = ReservedClient(client, self.ledger, run_id)
            else:
                clients[provider] = build_client(
                    provider, credentials, config.models.retry.request_timeout_s
                )
        if self.admission is None:
            self.admission = ProviderAdmission(
                config.models.rate_limits, config.pools.llm_concurrency
            )
        return LLMAdapter(
            models=config.models,
            profile=config.model_profile,
            prices=load_price_table(),
            clients=clients,
            events=EventLog(database),
            admission=self.admission,
            io_store=FileModelIOStore(runs_root),
            secrets=credentials.secret_values(),
        )


class AccountedAgent(AgentSystem):
    """Expose unknown paid exposure to the runner without changing SystemRun/frozen contracts."""

    def __init__(self, factory: AdapterFactory) -> None:
        super().__init__(adapter_factory=factory.for_run)
        self.ledger = factory.ledger

    def unobserved_spend_reserved_usd(self, run_id: str) -> float:
        """Unknown provider charges retained by the aggregate collection ledger."""
        return self.ledger.unobserved(run_id)


def recorded_code() -> dict[str, Any]:
    """Refuse recording against uncommitted source; result additions are not source changes."""
    code = GitRepo().state()
    if code.dirty:
        raise SpendError("commit source/model config before collecting recorded evidence")
    return {"commit": code.commit, "dirty": False}


async def comparison(profile: ModelProfile, resume: bool, ledger: SpendLedger) -> int:
    """Six actual cases on agent and baseline using the existing comparison runner."""
    recorded_code()
    plan = build_plan(
        EvalOptions(
            cases="dev",
            config="single_agent",
            systems=("agent", "baseline"),
            profile=profile,
            parallel=1,
            resume=resume,
            max_total_usd=CAP_USD,
        ),
        print,
    )
    factory = AdapterFactory(ledger)
    plan = replace(plan, systems=(AccountedAgent(factory), plan.systems[1]))
    outcome = await run_eval(plan)
    ledger.publish()
    return 0 if outcome.clean else 1


def smoke_adapter(factory: AdapterFactory, run_id: str) -> CLIAdapterFactory:
    """Bind a smoke run ID without adding parameters to the existing CLI adapter protocol."""

    def build(run_config: RunConfig, database: Database, runs_root: Path) -> LLMAdapter:
        return factory.for_run(run_config, database, runs_root, run_id)

    return build


async def matching_smokes(ledger: SpendLedger) -> int:
    """Actually rerun both S1.7 tasks on the pinned eval profile; retain the original records."""
    code = recorded_code()
    config = RunConfig.build(
        load_settings_for_eval("single_agent", profile=ModelProfile.EVAL),
        eval_config="single_agent",
    )
    factory = AdapterFactory(ledger)
    variant = smoke.Variant(
        "eval", Provider.OPENAI, "gpt-6-luna", "not_configurable", paid=True, remap=False
    )
    failures = 0
    for task in smoke.TASKS:
        destination = ROOT / "results" / "stage1" / "eval" / f"{task.task_id}.json"
        if destination.exists():
            old = json.loads(destination.read_text())
            if not (
                old.get("recorded") is True
                and old.get("profile") == "eval"
                and old.get("config_hash") == config.hash
                and old.get("code") == code
                and old.get("models") == ["openai/gpt-6-luna"]
                and all(
                    call.get("profile") == "eval" and not call.get("fallback_used")
                    for call in old.get("llm", {}).get("per_call", [])
                )
            ):
                raise SpendError(
                    "existing smoke does not match this source/config; never overwrite it"
                )
            pending = STATE / f"smoke-{task.task_id}.pending"
            if pending.exists():
                if pending.read_text().strip() != old["run"]["run_id"]:
                    raise SpendError("smoke intent conflicts with published evidence")
                pending.unlink()
            print(f"Keep matching eval smoke: {task.task_id}")
            failures += not old["run"]["verified"]
            continue
        snapshot = smoke.snapshot_for(task)
        run_id = f"run_s26_eval_{task.task_id.replace('-', '_')}_{uuid4().hex[:10]}"
        folder = STATE / "smoke"
        db_path = folder / "hypora.sqlite"
        # Durable intent refuses an ambiguous restart; the provider ledger separately keeps spend.
        intent = STATE / f"smoke-{task.task_id}.pending"
        intent.parent.mkdir(parents=True, exist_ok=True)
        with intent.open("x") as stream:
            stream.write(run_id + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        options = RunOptions(
            data=SNAPSHOTS_DIR / snapshot.file,
            goal=task.goal,
            target=task.target,
            db=db_path,
            runs_root=folder,
            budget_usd=SMOKE_RUN_BUDGET_USD,
            max_trials=4,
            max_wall_clock_s=1800,
            approve=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
            run_id=run_id,
        )

        result = await execute(
            options, out=print, adapter_factory=smoke_adapter(factory, run_id), run_config=config
        )
        database = Database(db_path)
        try:
            record = smoke.collect_record(
                database,
                folder,
                result.run_id,
                task=task,
                dataset=smoke.dataset_block(snapshot),
                variant=variant,
                context=smoke.RunContext(
                    profile="eval",
                    config_hash=config.hash,
                    budget_usd=SMOKE_RUN_BUDGET_USD,
                    code=code,
                ),
            )
        finally:
            database.close()
        record["paid_accounting"] = {
            "ledger": "results/stage2/paid-accounting.json",
            "unknown_reserved_usd": ledger.unobserved(run_id),
        }
        smoke.write_record(record, result.report_md)
        intent.unlink()
        ledger.publish()
        failures += not result.verified
    records = smoke.load_records("eval")
    summary = [
        "# Matching S1.7 tasks on pinned eval (S2.6)",
        "",
        "Actual eval-profile runs, not the old dev Luna remap.",
        "",
        "| Task | Status | Verified | Model | Input snapshot SHA-256 |",
        "|---|---|---|---|---|",
    ]
    summary += [
        f"| {r['task']['task_id']} | {r['run']['status']} | {r['run']['verified']} | "
        f"{', '.join(r['models'])} | {r['dataset']['sha256']} |"
        for r in records
    ]
    (ROOT / "results/stage1/eval/summary.md").write_text("\n".join(summary) + "\n")
    return 1 if failures else 0


@contextmanager
def collection_lock() -> Iterator[None]:
    """Only one collector publishes these primary/smoke records at once."""
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "collection.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SpendError("another S2.6 collector is already running") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_published_history(ledger: SpendLedger) -> None:
    """A new clone or missing journal cannot reset already-published paid exposure."""
    audit = ROOT / "results" / "stage2" / "paid-accounting.json"
    if not audit.exists():
        return
    if not ledger.path.exists():
        raise SpendError(
            "published paid history exists but durable ledger is missing; do not reset cap"
        )
    prior = json.loads(audit.read_text())
    current = {row["id"]: row for row in ledger.snapshot().attempts}
    for row in prior["attempts"]:
        if row["id"] not in current or (
            row["actual_usd"] is not None and row["actual_usd"] != current[row["id"]]["actual_usd"]
        ):
            raise SpendError(
                "durable ledger lost published paid history; inspect before collection"
            )


def main(argv: list[str] | None = None) -> int:
    """The owner-approved S2.6 collection entry point; no cap increase or model remap exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--profile", choices=("dev", "eval"), required=True)
    run.add_argument("--resume", action="store_true")
    commands.add_parser("smoke")
    commands.add_parser("spend")
    args = parser.parse_args(argv)
    ledger = SpendLedger()
    try:
        if args.command == "spend":
            print(ledger.publish())
            return 0
        with collection_lock():
            check_published_history(ledger)
            if args.command == "smoke":
                return anyio.run(matching_smokes, ledger)
            return anyio.run(comparison, ModelProfile(args.profile), args.resume, ledger)
    except (SpendError, EvalRefused) as exc:
        print(f"Collection refused: {exc}")
        return 2
    finally:
        try:
            ledger.publish()
        except SpendError:
            print("Paid ledger needs inspection; no reservation released; audit not regenerated.")


if __name__ == "__main__":
    raise SystemExit(main())
