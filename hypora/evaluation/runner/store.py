"""The results store (spec §19.6; PLAN.md S2.4): where records live, and the rules for writing them.

    results/{commit_sha}/{config}/{system}/{case_id}.json             a primary run
    results/{commit_sha}/{config}/repeats/{system}/{case_id}/{n}.json the repeat sample
    results/{commit_sha}/{config}/summary.md, summary.json, regressions.json

The rules (docs/notes/eval-runner.md):
- A record describes one commit, so a dirty tree is refused (`results/` itself doesn't count, since
  records are what it adds). `--allow-dirty` writes to the git-ignored `results/raw/`, which can't
  be committed by accident and is never a regression baseline.
- A finished primary record is never replaced: write-once, enforced by the file system. A
  `harness_error` record is not a measurement, so a later run replaces it.
- `{config}` is the config's name, or `{name}.{profile}` when `--profile` overrides the profile the
  config states, so one commit can hold both profiles (D-016) without their records colliding.
"""

import contextlib
import fcntl
import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import uuid4

from pydantic import JsonValue, ValidationError

from hypora.core.schemas.events import ModelProfile
from hypora.evaluation.runner.records import CaseRecord, PendingRun
from hypora.storage.snapshots import REPO_ROOT

RESULTS_ROOT: Final = REPO_ROOT / "results"
RAW_DIR: Final = "raw"
"""Under `results/` and git-ignored (PLAN.md S0.1)."""
REPEATS_DIR: Final = "repeats"
ATTEMPTS_DIR: Final = "attempts"
PENDING_DIR: Final = "pending"
SUMMARY_MD: Final = "summary.md"
SUMMARY_JSON: Final = "summary.json"
REGRESSIONS_JSON: Final = "regressions.json"
EVAL_RESULTS_PATHSPEC: Final = ":(exclude)results"


class EvalRefused(Exception):
    """An eval that must not start, or a record that must not be written; says what to do."""


class RecordExists(EvalRefused):
    """A finished primary record is already at that path."""


# --- Which code this is --------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeState:
    """The checked-out commit, and whether the working tree differs from it (results aside)."""

    commit: str
    dirty: bool


class GitRepo:
    """The read-only git questions the runner asks."""

    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.root = root

    def _git(self, *args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=False
        )
        if done.returncode != 0:
            raise EvalRefused(f"git {args[0]} failed in {self.root}: {done.stderr.strip()}")
        return done.stdout.strip()

    def state(self) -> CodeState:
        """HEAD and its dirtiness. Untracked files count, because the code can import them."""
        status = self._git("status", "--porcelain", "--", ".", EVAL_RESULTS_PATHSPEC)
        return CodeState(self._git("rev-parse", "HEAD"), bool(status))

    def ancestors(self, start: str, limit: int) -> list[str]:
        """`start` and its first-parent history, nearest first; empty when `start` doesn't exist
        (the parent of a root commit, or a shallow clone)."""
        try:
            listing = self._git("rev-list", "--first-parent", f"--max-count={limit}", start)
        except EvalRefused:
            return []
        return listing.split()


def results_segment(config: str, profile: ModelProfile, config_profile: ModelProfile) -> str:
    """The `{config}` level of a result path: the profile joins it only when it was overridden."""
    return config if profile is config_profile else f"{config}.{profile.value}"


# --- Reading and writing -----------------------------------------------------------------------


@dataclass(frozen=True)
class ResultsStore:
    """One commit and config's results directory."""

    root: Path
    commit: str
    segment: str
    dirty: bool = False

    @property
    def directory(self) -> Path:
        """Where this eval's records go: dirty runs live under the git-ignored `raw/`."""
        if self.dirty:
            return self.root / RAW_DIR / f"{self.commit}-dirty" / self.segment
        return self.root / self.commit / self.segment

    def primary_path(self, system: str, case_id: str) -> Path:
        """`{system}/{case_id}.json`."""
        return self.directory / system / f"{case_id}.json"

    def repeat_path(self, system: str, case_id: str, attempt: int) -> Path:
        """`repeats/{system}/{case_id}/{attempt}.json`: never read as a primary run."""
        return self.directory / REPEATS_DIR / system / case_id / f"{attempt}.json"

    def path_for(self, record: CaseRecord) -> Path:
        """Where `record` belongs."""
        if record.kind.value == "primary":
            return self.primary_path(record.system, record.case_id)
        return self.repeat_path(record.system, record.case_id, record.attempt)

    def attempt_path(self, record: CaseRecord) -> Path:
        """Immutable history of a replaced run; never a new primary or repeat observation."""
        return (
            self.directory
            / ATTEMPTS_DIR
            / record.kind.value
            / record.system
            / record.case_id
            / f"{record.run_id}.json"
        )

    def read(self, path: Path, *, archived: bool = False) -> CaseRecord | None:
        """The record at `path`, or None. A corrupt record is refused, never taken for missing."""
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            record = CaseRecord.model_validate_json(content)
        except ValidationError as exc:
            raise EvalRefused(
                f"{path} is not a valid record ({exc.error_count()} problem(s)); fix or delete it"
            ) from None
        provenance = record.provenance
        valid_segments = {provenance.config, f"{provenance.config}.{provenance.profile.value}"}
        if (
            path != (self.attempt_path(record) if archived else self.path_for(record))
            or provenance.commit_sha != self.commit
            or provenance.dirty != self.dirty
            or self.segment not in valid_segments
        ):
            raise EvalRefused(
                f"{path} has record identity or provenance inconsistent with its path"
            )
        return record

    def write(self, document: JsonValue, path: Path) -> None:
        """Publish `document` at `path` in one step.

        `os.link` fails if the path exists, so two runs can't both finish the same record. Only a
        `harness_error` record, or anything in a dirty run's scratch area, may be replaced.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.read(path)
                if existing is not None and existing.finished and not self.dirty:
                    raise RecordExists(
                        f"{path} is a finished record and is not replaced; delete it yourself to "
                        "run this case again"
                    ) from None
                if existing is not None:
                    self.archive(path, existing)
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def archive(self, source: Path, record: CaseRecord) -> None:
        """Link the original bytes before replacement. A crash/retry reuses the same archive.

        The runner holds `exclusive()` across publication, so no other eval can replace source
        between this read and link. Refuse an inconsistent duplicate instead of losing evidence.
        """
        target = self.attempt_path(record)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except FileExistsError:
                if target.read_bytes() != source.read_bytes():
                    raise EvalRefused(
                        f"{target} conflicts with the prior attempt; resume refused"
                    ) from None
        except OSError as exc:
            raise EvalRefused(
                f"could not archive the prior attempt at {target}: {exc.strerror}; resume refused"
            ) from None

    def primary_records(self) -> dict[tuple[str, str], CaseRecord]:
        """Every primary record on disk, by (system, case ID)."""
        records: dict[tuple[str, str], CaseRecord] = {}
        if not self.directory.is_dir():
            return records
        for folder in sorted(p for p in self.directory.iterdir() if p.is_dir()):
            if folder.name in {REPEATS_DIR, ATTEMPTS_DIR, PENDING_DIR}:
                continue
            for path in sorted(folder.glob("*.json")):
                record = self.read(path)
                if record is not None:
                    records[record.system, record.case_id] = record
        return records

    def repeat_records(self) -> list[CaseRecord]:
        """Every record of the repeat sample on disk."""
        folder = self.directory / REPEATS_DIR
        if not folder.is_dir():
            return []
        records = (self.read(path) for path in sorted(folder.glob("*/*/*.json")))
        return [record for record in records if record is not None]

    def attempt_records(self) -> list[CaseRecord]:
        """Replaced runs, kept for complete recovery-cost and timing accounting."""
        folder = self.directory / ATTEMPTS_DIR
        if not folder.is_dir():
            return []
        records = (self.read(path, archived=True) for path in sorted(folder.glob("*/*/*/*.json")))
        return [record for record in records if record is not None]

    def begin_pending(self, receipt: PendingRun) -> None:
        """Atomically reserve a run identity and full budget before dispatching any model call."""
        folder = self.directory / PENDING_DIR
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{receipt.run_id}.json"
        temporary = folder / f".{receipt.run_id}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise EvalRefused(
                    f"run {receipt.run_id} already has an accounting receipt"
                ) from None
        finally:
            temporary.unlink(missing_ok=True)

    def finish_pending(self, run_id: str) -> None:
        """Only call after a complete or conservatively reserved partial record is published."""
        (self.directory / PENDING_DIR / f"{run_id}.json").unlink(missing_ok=True)

    def refuse_unreconciled_pending(self) -> None:
        """Reconcile a crash after final publication; otherwise refuse before any further spend.

        This is accounting only, not controller recovery. An abrupt death before publication
        requires inspecting the run's event database and preserving its usage before removal.
        """
        folder = self.directory / PENDING_DIR
        paths = sorted(folder.glob("*.json")) if folder.is_dir() else []
        if not paths:
            return
        records = (
            *self.attempt_records(),
            *self.repeat_records(),
            *self.primary_records().values(),
        )
        by_id = {record.run_id: record for record in records}
        for path in paths:
            try:
                receipt = PendingRun.model_validate_json(path.read_bytes())
                if path.name != f"{receipt.run_id}.json":
                    raise EvalRefused(f"pending receipt identity does not match its path at {path}")
            except (OSError, ValidationError):
                raise EvalRefused(
                    f"invalid pending accounting receipt at {path}; inspect it before resuming"
                ) from None
            record = by_id.get(receipt.run_id)
            if record is not None and (
                (record.case_id, record.system, record.kind, record.attempt)
                == (receipt.case_id, receipt.system, receipt.kind, receipt.attempt)
                and record.inputs.case_sha256 == receipt.case_sha256
                and record.provenance.run_config_hash == receipt.run_config_hash
            ):
                self.finish_pending(receipt.run_id)
                continue
            raise EvalRefused(
                f"unreconciled accounting receipt for {receipt.system}/{receipt.case_id}, "
                f"run {receipt.run_id}; up to ${receipt.budget_usd:.2f} may be charged. "
                "No new runs started. Inspect this run's state/<run_id>.sqlite event ledger, "
                "preserve its usage as a partial result, then remove the receipt only after "
                "accounting for known and unknown charges."
            )

    def write_derived(self, files: Mapping[str, str]) -> None:
        """Reports are regenerated from the records, so these are simply replaced."""
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            temporary = self.directory / f".{name}.{uuid4().hex}.tmp"
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, self.directory / name)

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Only one eval records here at a time. The lock is on the directory, so it leaves no
        file behind in what gets committed, and the kernel drops it if the process dies."""
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise EvalRefused(
                    f"another eval is recording in {self.directory}; wait for it to finish"
                ) from None
            yield
        finally:
            with contextlib.suppress(OSError):
                os.close(descriptor)


__all__ = [
    "REGRESSIONS_JSON",
    "RESULTS_ROOT",
    "SUMMARY_JSON",
    "SUMMARY_MD",
    "CodeState",
    "EvalRefused",
    "GitRepo",
    "RecordExists",
    "ResultsStore",
    "results_segment",
]
