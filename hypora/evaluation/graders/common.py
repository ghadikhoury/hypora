"""What every grader shares: check constructors, fail-closed guarding, and the run's evidence.

A check that can't run is a failed check. `guarded` turns any exception into a FAILED result, and
`MissingEvidence` is how a grader says what was missing. A check's `detail` goes into result records
in a public repository, so exception text is only used when this package wrote it: pandas and
scikit-learn errors can quote hidden values, and sandbox logs are written by trial code.
"""

import io
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property

import pandas as pd  # type: ignore[import-untyped]

from hypora.core.schemas import AnyEvent, ExperimentContract
from hypora.evaluation.case import Case, DecisionClass
from hypora.evaluation.hidden_split import HiddenSplit
from hypora.evaluation.system import CheckStatus, FinalModel, GraderCheck, SystemRun
from hypora.execution.harness.data import PreparedData, prepare_data
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import RecordNotFound, StoredRun, load_run

log = logging.getLogger(__name__)


class MissingEvidence(Exception):
    """A required check can't be decided. The message goes into the check, so it must be ours."""

    def __init__(self, detail: str, refs: Iterable[str] = ()) -> None:
        super().__init__(detail)
        self.detail = detail
        self.refs = tuple(refs)


def passed(name: str, detail: str, refs: Iterable[str] = ()) -> GraderCheck:
    """A check that passed."""
    return GraderCheck(
        name=name, status=CheckStatus.PASSED, detail=detail, evidence_refs=tuple(refs)
    )


def failed(name: str, detail: str, refs: Iterable[str] = ()) -> GraderCheck:
    """A check that failed, or could not be decided."""
    return GraderCheck(
        name=name, status=CheckStatus.FAILED, detail=detail, evidence_refs=tuple(refs)
    )


def not_applicable(name: str, detail: str) -> GraderCheck:
    """A check this case truly does not call for. Never use it for missing evidence."""
    return GraderCheck(name=name, status=CheckStatus.NOT_APPLICABLE, detail=detail)


def verdict(name: str, problems: list[str], success: str, refs: Iterable[str] = ()) -> GraderCheck:
    """PASSED with `success`, or FAILED listing every problem."""
    if problems:
        return failed(name, "; ".join(problems), refs)
    return passed(name, success, refs)


def guarded(name: str, check: Callable[[], GraderCheck]) -> GraderCheck:
    """Run one check. Whatever goes wrong, the result is a FAILED check, never a pass or a crash."""
    try:
        return check()
    except MissingEvidence as exc:
        return failed(name, exc.detail, exc.refs)
    except Exception as exc:
        # Only the type: the message may quote data (see the module docstring).
        log.warning("grader check %s could not run: %s", name, type(exc).__name__)
        return failed(name, f"the check could not run: {type(exc).__name__}")


def event_ref(event: AnyEvent) -> str:
    """Where one event is: `event:<run_id>:<seq>`."""
    return f"event:{event.run_id}:{event.seq}"


@dataclass(frozen=True)
class RunLog:
    """A run's durable records: its `runs` row and its event log, in sequence order."""

    database: Database
    stored: StoredRun
    events: tuple[AnyEvent, ...]

    @property
    def ref(self) -> str:
        """Where the whole log is."""
        return f"events:{self.stored.run_id}"


def load_run_log(run: SystemRun) -> RunLog:
    """Read a run's records. A run with no row or no events has no evidence of anything."""
    if not run.database.is_file():  # opening a missing path would create an empty database
        raise MissingEvidence("the run's database does not exist")
    database = Database(run.database)
    try:
        stored = load_run(database, run.run_id)
        events = tuple(EventLog(database).read(run.run_id))
    except RecordNotFound:
        database.close()
        raise MissingEvidence(f"the database has no run {run.run_id}") from None
    except Exception as exc:
        database.close()
        raise MissingEvidence(f"the run's records can't be read: {type(exc).__name__}") from None
    if not events:
        database.close()
        raise MissingEvidence(f"run {run.run_id} has an empty event log")
    return RunLog(database, stored, events)


class GradingContext:
    """One grade's inputs, with the expensive evidence read at most once."""

    def __init__(self, case: Case, split: HiddenSplit, run: SystemRun) -> None:
        self.case = case
        self.split = split
        self.run = run
        self._log: RunLog | None = None
        self._log_problem: MissingEvidence | None = None

    def run_log(self) -> RunLog:
        """The run's records, or MissingEvidence for every check that needs them."""
        if self._log is None and self._log_problem is None:
            try:
                self._log = load_run_log(self.run)
            except MissingEvidence as exc:
                self._log_problem = exc
        if self._log_problem is not None:
            raise self._log_problem
        assert self._log is not None
        return self._log

    def close(self) -> None:
        """Release the database handle."""
        if self._log is not None:
            self._log.database.close()

    @property
    def expects_no_model(self) -> bool:
        """Whether the right outcome of this case is having no model at all."""
        return self.case.expected_decision_class is DecisionClass.INFEASIBLE

    def final_model(self) -> FinalModel:
        """The run's final model, or MissingEvidence: only an infeasible case may have none."""
        final = self.run.final_model
        if final is None:
            raise MissingEvidence("the run has no final model")
        return final

    @property
    def contract(self) -> ExperimentContract:
        """The contract of the run's final model."""
        return self.final_model().contract

    @cached_property
    def visible(self) -> pd.DataFrame:
        """The rows the system received, from the hash-checked file."""
        return pd.read_parquet(io.BytesIO(self.split.visible_bytes()), engine="pyarrow")

    @cached_property
    def prepared(self) -> PreparedData:
        """The split the harness really used: its own code, on the visible rows and the contract.

        Row IDs are positions in `visible` (`apply_contract` never reorders). Deterministic for a
        given contract seed, so this needs nothing a system reports about its split.
        """
        try:
            return prepare_data(self.visible, self.contract)
        except MissingEvidence:
            raise
        except Exception as exc:
            raise MissingEvidence(
                f"the contract's split can't be rebuilt on the visible rows: {type(exc).__name__}",
                [f"contract:{self.contract.hash}"],
            ) from None


__all__ = [
    "GradingContext",
    "MissingEvidence",
    "RunLog",
    "event_ref",
    "failed",
    "guarded",
    "load_run_log",
    "not_applicable",
    "passed",
    "verdict",
]
