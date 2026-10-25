"""The six S1.4 tools (spec §12; PLAN.md S1.4). Docstrings are the descriptions the LLM reads.

Every description says what the tool does, when to use it, when not to, what it returns, and an
example (spec §12 design rules). Parameters carry their own descriptions and constraints, so the
input schema is the documentation. Tools raise `ToolFailure` for any problem the agent can act on;
the server's call boundary (`server.py`) emits events, applies the allowlist and the data
firewall, and turns failures into structured `ToolError` results.
"""

import ast
import contextvars
import errno
import hashlib
import json
import math
import re
import secrets
import statistics
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Final

import yaml  # type: ignore[import-untyped]
from pydantic import Field, JsonValue, ValidationError
from sqlalchemy import select

from hypora.core.policy.firewall import DataFirewall
from hypora.core.schemas import (
    TRIAL_DIR_TEMPLATE,
    ApprovalDecision,
    DatasetProfile,
    ExperimentContract,
    Family,
    IdempotencyKey,
    Identifier,
    JobLimits,
    JobState,
    MetricName,
    Operation,
    PinnedModelRef,
    SubmitExperimentRequest,
    TaskType,
    TrialFile,
)
from hypora.core.schemas.events import PolicyRule
from hypora.core.schemas.trials import TrialDesign
from hypora.execution.protocol import experiments as protocol
from hypora.execution.sandbox.outputs import read_output_bytes
from hypora.mcp_server import errors
from hypora.mcp_server.models import (
    ArtifactRef,
    ClassScore,
    ExperimentError,
    ExperimentLogs,
    ExperimentMetrics,
    ExperimentStatus,
    FoldScore,
    GroupSummary,
    LogLine,
    MetricSummary,
    SubmitDefaults,
    SubmitResult,
    TrialFilesResult,
    WrittenFile,
)
from hypora.mcp_server.session import Session
from hypora.mcp_server.trialfs import existing_files, write_file
from hypora.profiling.tabular import profile_tabular
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.jobs import TERMINAL_STATES, JobNotFound, JobRecord, new_job_id
from hypora.storage.models import Job
from hypora.storage.records import (
    RecordNotFound,
    StoredTrial,
    load_approval,
    load_experiment_contract,
    load_trial,
    set_trial_status,
    upsert_trial,
)

MAX_PIPELINE_BYTES: Final = 1024 * 1024
MAX_LOCK_BYTES: Final = 64 * 1024
TRIAL_FILES: Final = ("trial.yaml", "pipeline.py", "requirements.lock")
WITHHELD_LINE: Final = "[line withheld by the data firewall: it showed dataset values]"
_SANDBOX_LOCK: Final = Path(__file__).parents[1] / "execution/sandbox/requirements-tabular.lock"
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([A-Za-z0-9._+!-]+)\s*$")
_PROFILED_TASKS = frozenset(
    {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION}
)


@dataclass
class CallNotes:
    """What a tool tells the call boundary about one call (set per call by the server)."""

    withheld: dict[str, int] = field(default_factory=dict)

    def add_withheld(self, counts: dict[str, int]) -> None:
        for rule, count in counts.items():
            self.withheld[rule] = self.withheld.get(rule, 0) + count


CURRENT_CALL: contextvars.ContextVar[CallNotes] = contextvars.ContextVar("hypora_mcp_call")


def _sandbox_packages() -> dict[str, str]:
    """`name -> version` pinned in the tabular sandbox image (the only installable set)."""
    packages = {}
    for line in _SANDBOX_LOCK.read_text(encoding="utf-8").splitlines():
        match = _REQUIREMENT.match(line.split(" \\")[0].split("#")[0])
        if match:
            packages[match[1].lower().replace("_", "-")] = match[2]
    return packages


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class HyporaTools:
    """Tool implementations bound to one session. Methods are registered as MCP tools.

    `HyporaTools(None)` has no session: it only lists the tool contract (docs/mcp/tools.json).
    """

    def __init__(self, session: Session | None) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._profiles: dict[tuple[str, str | None, str | None], DatasetProfile] = {}
        self._packages = _sandbox_packages()

    @property
    def session(self) -> Session:
        if self._session is None:
            raise RuntimeError("these tools were built for listing only (no session)")
        return self._session

    # --- shared checks ----------------------------------------------------------------------------

    def _contract(self) -> ExperimentContract | None:
        return self.session.contract()

    def _trial(self, trial_id: str) -> StoredTrial:
        try:
            trial = load_trial(self.session.database, trial_id)
        except RecordNotFound:
            trial = None
        if trial is None or trial.run_id != self.session.run_id:
            raise errors.missing(
                "TRIAL_NOT_FOUND",
                f"This run has no trial {trial_id!r}.",
                "Create a trial with write_trial_files (omit trial_id) and use the trial_id "
                "it returns.",
                trial_id=trial_id,
            )
        return trial

    def _job(self, job_id: str) -> JobRecord:
        try:
            job = self.session.queue.get(job_id)
        except JobNotFound:
            job = None
        if job is None or job.run_id != self.session.run_id:
            raise errors.missing(
                "EXPERIMENT_NOT_FOUND",
                f"This run has no experiment job {job_id!r}.",
                "Use the job_id returned by submit_experiment.",
                job_id=job_id,
            )
        return job

    def _firewall(self) -> DataFirewall | None:
        return self.session.firewall() if self.session.config.defenses.firewall else None

    def _screen(self, lines: Sequence[str]) -> tuple[list[str], int]:
        """Withhold trial-produced lines that show dataset values (D-018 untrusted-text rule)."""
        firewall = self._firewall()
        if firewall is None:
            return list(lines), 0
        screened, counts = firewall.screen_untrusted(lines)
        if counts:
            notes = CURRENT_CALL.get(None)
            if notes is not None:
                notes.add_withheld(counts)
        return [WITHHELD_LINE if line is None else line for line in screened], sum(counts.values())

    def _trial_dir_parts(self, trial_id: str) -> tuple[str, ...]:
        return (self.session.run_id, "trials", trial_id)

    def _trial_dir(self, trial_id: str) -> Path:
        return self.session.runs_root.joinpath(*self._trial_dir_parts(trial_id))

    def _submit_defaults(self, contract: ExperimentContract | None) -> SubmitDefaults | None:
        if contract is None:
            return None
        return SubmitDefaults(
            run_id=contract.run_id,
            family=contract.family,
            contract_version=contract.version,
            approval_ref=contract.approval_id,
            limits=contract.default_job_limits,
        )

    # --- inspect_dataset --------------------------------------------------------------------------

    def inspect_dataset(
        self,
        dataset_id: Annotated[
            Identifier | None,
            Field(
                description="An authorized dataset ID, e.g. 'ds_main'. Omit for the dataset "
                "the run's experiment contract governs."
            ),
        ] = None,
        target: Annotated[
            str | None,
            Field(
                description="Column to treat as the label for target-aware statistics. Omit to "
                "use the contract's target (or, before a contract exists, no target)."
            ),
        ] = None,
        task_type: Annotated[
            TaskType | None,
            Field(
                description="Required with `target`: binary/multiclass classification or "
                "regression."
            ),
        ] = None,
    ) -> DatasetProfile:
        """Summarize an authorized tabular dataset without revealing any of its rows.

        Use it first, to learn the dataset's shape before designing a trial: column types,
        missing values, value ranges, the target's class balance, ID-like columns, candidate
        group keys (repeated subjects or batches), timestamps, and leakage candidates (single
        columns that predict the target suspiciously well; a signal to review, not proof).

        Do not use it to look at rows, samples, or individual values: no tool returns them (the
        data firewall). Categorical columns list at most their five most frequent labels.

        Returns a DatasetProfile. `tabular.leakage_screen_skipped_reason`, when set, means
        leakage was not screened; an empty `leakage_candidates` list then proves nothing.
        Profiles are cached: repeating the same call is cheap.

        Example: inspect_dataset(dataset_id="ds_main")
        """
        session = self.session
        dataset_id = dataset_id or session.config.experiment_dataset
        path = session.dataset_path(dataset_id)
        if path is None:
            raise errors.missing(
                "DATASET_NOT_FOUND",
                f"{dataset_id!r} is not an authorized dataset of this session.",
                "Omit dataset_id to inspect the run's dataset, or use an ID you were given.",
                authorized=", ".join(sorted(session.config.datasets)),
            )
        if (target is None) != (task_type is None):
            if target is None:
                raise errors.invalid(
                    "TARGET_REQUIRED",
                    "task_type was given without target.",
                    "Pass both target and task_type, or neither.",
                )
            raise errors.invalid(
                "TASK_TYPE_REQUIRED",
                "target was given without task_type.",
                "Pass both target and task_type, or neither.",
            )
        contract = self._contract()
        if (
            target is None
            and contract is not None
            and dataset_id == session.config.experiment_dataset
        ):
            target, task_type = contract.target, contract.task_type
        if task_type is not None and task_type not in _PROFILED_TASKS:
            raise errors.invalid(
                "TASK_TYPE_UNSUPPORTED",
                f"Profiling supports {sorted(t.value for t in _PROFILED_TASKS)}.",
                "Choose one of the supported task types.",
            )
        frame = session.frame(dataset_id)
        if target is not None and target not in frame.columns:
            raise errors.invalid(
                "UNKNOWN_COLUMN",
                f"The dataset has no column {target!r}.",
                "Call inspect_dataset without a target to list the columns.",
            )
        key = (dataset_id, target, task_type.value if task_type else None)
        with self._lock:
            cached = self._profiles.get(key)
        if cached is not None:
            return cached
        profile = profile_tabular(
            frame,
            run_id=session.run_id,
            dataset_ref=f"dataset:{dataset_id}",
            target=target,
            task_type=task_type,
        )
        profile.fingerprint = _sha256(path.read_bytes())
        with self._lock:
            self._profiles[key] = profile
        return profile

    # --- write_trial_files ------------------------------------------------------------------------

    def write_trial_files(
        self,
        design: Annotated[
            TrialDesign | None,
            Field(
                description="The trial's hypothesis and plan, saved as trial.yaml. Required when "
                "creating a trial. For a fit trial: kind 'baseline' or 'candidate', operation "
                "'fit'."
            ),
        ] = None,
        pipeline_py: Annotated[
            str | None,
            Field(
                description="Full source of pipeline.py, defining fit(train, ctx) -> model and "
                "predict(model, X) -> Predictions (see the tool description). Required when "
                "creating a trial. At most 1 MiB.",
            ),
        ] = None,
        requirements_lock: Annotated[
            str | None,
            Field(
                description="Optional requirements.lock: 'name==version' lines, each already "
                "installed in the sandbox image. New packages need user approval."
            ),
        ] = None,
        trial_id: Annotated[
            Identifier | None,
            Field(
                description="Omit to create a new trial; pass the trial_id from an earlier call "
                "to replace that trial's files, e.g. 'trl_4f2a9c1e'."
            ),
        ] = None,
    ) -> TrialFilesResult:
        """Create a trial, or replace its files, in the run's own trial directory.

        A trial tests ONE hypothesis. Its pipeline.py is model code that the platform runs in a
        sandbox: `fit(train, ctx)` receives `train.X` / `train.y` (pandas; training rows only)
        and must do all learned preprocessing (imputation, scaling, encoding) inside fit;
        `predict(model, X)` returns `runtime.Predictions(values, probabilities, classes)`, with
        class probabilities in sorted-label order for classification. Import `Predictions` from
        `runtime`. The platform, never this code, splits the data and computes the metrics.

        Use it to start a trial (omit trial_id), or to fix a trial after reading its logs
        (pass trial_id and only the files that change). Then call submit_experiment.

        Do not use it to read files or to write anywhere but this trial: paths are fixed. Do not
        print data from pipeline.py; output that shows dataset values is withheld from logs.

        Returns the trial_id, the files written (SHA-256, size), the files the trial now has,
        and `submit_defaults`: the values submit_experiment uses for omitted arguments.

        Example: write_trial_files(design={"kind": "baseline", "operation": "fit",
        "hypothesis": "A majority-class predictor sets the floor", "model_family": "dummy",
        "expected_effect": "f1_macro near 0.4", "estimated_cost": {"compute_seconds": 30}},
        pipeline_py="...")
        """
        session = self.session
        created = trial_id is None
        if created:
            if design is None or pipeline_py is None:
                raise errors.invalid(
                    "TRIAL_INCOMPLETE",
                    "A new trial needs both design and pipeline_py.",
                    "Pass design and pipeline_py (requirements_lock is optional).",
                )
            trial_id = self._new_trial_id()
        else:
            assert trial_id is not None
            self._trial(trial_id)
            if design is None and pipeline_py is None and requirements_lock is None:
                raise errors.invalid(
                    "NOTHING_TO_WRITE",
                    "No file content was given.",
                    "Pass at least one of design, pipeline_py, requirements_lock.",
                )
        files: dict[str, bytes] = {}
        trial_file: TrialFile | None = None
        if design is not None:
            trial_file = TrialFile.model_validate({"trial_id": trial_id, **design.model_dump()})
            files["trial.yaml"] = yaml.safe_dump(
                trial_file.model_dump(mode="json"), sort_keys=False
            ).encode()
        if pipeline_py is not None:
            files["pipeline.py"] = self._checked_pipeline(pipeline_py)
        if requirements_lock is not None:
            files["requirements.lock"] = self._checked_lock(requirements_lock)

        parts = self._trial_dir_parts(trial_id)
        directory = TRIAL_DIR_TEMPLATE.format(run_id=session.run_id, trial_id=trial_id)
        try:
            for name, content in files.items():
                write_file(session.runs_root, parts, name, content)
        except OSError as exc:
            if exc.errno not in (errno.ELOOP, errno.ENOTDIR):
                raise
            raise errors.policy(
                "UNSAFE_TRIAL_DIR",
                "The trial directory contains a link; nothing was written.",
                "Report this to the user: platform directories must never contain links.",
                decided_by=PolicyRule.CONTRACT_SCOPE,
            ) from None
        if trial_file is not None:
            upsert_trial(session.database, session.run_id, trial_file, directory)
        contract = self._contract()
        return TrialFilesResult.model_validate(
            {
                "trial_id": trial_id,
                "created": created,
                "trial_dir": directory,
                "files": [
                    WrittenFile.model_validate(
                        {"name": name, "sha256": _sha256(content), "bytes": len(content)}
                    )
                    for name, content in files.items()
                ],
                "present": existing_files(session.runs_root, parts, TRIAL_FILES),
                "submit_defaults": self._submit_defaults(contract),
                "next_step": self._after_write(trial_id, contract),
            }
        )

    def _after_write(self, trial_id: str, contract: ExperimentContract | None) -> str:
        if contract is None:
            return "Wait: the run has no approved experiment contract yet."
        return (
            f"Call submit_experiment(trial_id={trial_id!r}, "
            f"idempotency_key='{self.session.run_id}:<new unique id>')."
        )

    def _new_trial_id(self) -> str:
        for _ in range(20):
            candidate = f"trl_{secrets.token_hex(4)}"
            try:
                load_trial(self.session.database, candidate)
            except RecordNotFound:
                return candidate
        raise RuntimeError("could not allocate a trial ID")

    def _checked_pipeline(self, source: str) -> bytes:
        content = source.encode("utf-8")
        if len(content) > MAX_PIPELINE_BYTES:
            raise errors.invalid(
                "PIPELINE_TOO_LARGE",
                f"pipeline.py is {len(content)} bytes (limit {MAX_PIPELINE_BYTES}).",
                "Shorten pipeline.py; generated code should be compact.",
            )
        try:
            module = ast.parse(source, filename="pipeline.py")  # parsed, never executed here
        except SyntaxError as exc:
            raise errors.invalid(
                "PIPELINE_SYNTAX_ERROR",
                f"pipeline.py line {exc.lineno}: {exc.msg}.",
                "Fix the syntax error and write pipeline.py again.",
                line=exc.lineno or 0,
            ) from None
        defined = {
            node.name
            for node in module.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        missing = sorted({"fit", "predict"} - defined)
        if missing:
            raise errors.invalid(
                "PIPELINE_INTERFACE",
                f"pipeline.py must define top-level functions fit and predict; missing {missing}.",
                "Define def fit(train, ctx) and def predict(model, X) at module level.",
            )
        return content

    def _checked_lock(self, text: str) -> bytes:
        content = text.encode("utf-8")
        if len(content) > MAX_LOCK_BYTES:
            raise errors.invalid(
                "LOCK_TOO_LARGE",
                f"requirements.lock is {len(content)} bytes (limit {MAX_LOCK_BYTES}).",
                "List only the packages the pipeline imports.",
            )
        unapproved = []
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            match = _REQUIREMENT.match(line)
            name = match[1].lower().replace("_", "-") if match else line
            if match is None or self._packages.get(name) != match[2]:
                unapproved.append(line[:80])
        if unapproved:
            raise errors.policy(
                "DEPENDENCY_NOT_APPROVED",
                f"Not installed in the sandbox image at that version: {unapproved[:5]}.",
                "Use only the preinstalled libraries (pandas, numpy, scikit-learn, xgboost, "
                "lightgbm, optuna) at their pinned versions; new packages need user approval.",
                decided_by=PolicyRule.APPROVAL_STATE,
            )
        return content

    # --- submit_experiment ------------------------------------------------------------------------

    def submit_experiment(
        self,
        trial_id: Annotated[
            Identifier, Field(description="The trial to run, from write_trial_files.")
        ],
        idempotency_key: Annotated[
            IdempotencyKey,
            Field(
                description="'{run_id}:{unique id}', e.g. 'run_104:sub_0001'. Use a NEW key for "
                "every new submission; resend the SAME key only to retry a submission whose "
                "reply you did not receive."
            ),
        ],
        operation: Annotated[
            Operation, Field(description="Only 'fit' is supported in this version.")
        ] = Operation.FIT,
        family: Annotated[
            Family, Field(description="Only 'tabular' is supported in this version.")
        ] = Family.TABULAR,
        run_id: Annotated[
            Identifier | None, Field(description="Omit; defaults to this session's run.")
        ] = None,
        contract_version: Annotated[
            int | None,
            Field(ge=1, description="Omit to use the current contract; if given, it must match."),
        ] = None,
        approval_ref: Annotated[
            Identifier | None,
            Field(description="Omit to use the contract's approval; if given, it must match."),
        ] = None,
        model_ref: Annotated[
            PinnedModelRef | None, Field(description="Pretrained weights; not used by 'fit'.")
        ] = None,
        config: Annotated[
            dict[str, JsonValue] | None,
            Field(description="Passed to fit/predict as ctx.config, e.g. {'max_depth': 6}."),
        ] = None,
        limits: Annotated[
            JobLimits | None,
            Field(
                description="Sandbox limits per stage; omit for the contract's defaults. They "
                "can be lowered but never raised above the contract's defaults."
            ),
        ] = None,
    ) -> SubmitResult:
        """Queue one trial for platform-run validation; returns a job_id at once.

        The platform freezes the trial's current files, then (in a sandbox, without network)
        fits pipeline.py on each validation fold of the contract's split and scores the
        predictions itself. Held-out test labels are never used here.

        Use it after write_trial_files, and again after fixing a failed trial (with a NEW
        idempotency_key). Then poll get_experiment_status until `finished` is true.

        Do not use it to change the split, metric, features, or budget: they come from the
        approved contract. Limits above the contract's are refused; only the user can extend
        them.

        Returns the job_id, its state (normally QUEUED), and whether the key was a duplicate.

        Example: submit_experiment(trial_id="trl_4f2a9c1e", idempotency_key="run_104:sub_0001")
        """
        session = self.session
        if run_id is not None and run_id != session.run_id:
            raise errors.policy(
                "RUN_SCOPE_VIOLATION",
                f"This session belongs to run {session.run_id!r}, not {run_id!r}.",
                "Omit run_id.",
                decided_by=PolicyRule.CONTRACT_SCOPE,
            )
        contract = self._contract()
        if contract is None:
            raise errors.missing(
                "NO_CONTRACT",
                "The run has no approved experiment contract yet.",
                "Wait for the plan to be approved; experiments need an approved contract.",
            )
        if contract_version is not None and contract_version != contract.version:
            raise errors.invalid(
                "CONTRACT_VERSION_MISMATCH",
                f"The current contract is version {contract.version}, not {contract_version}.",
                "Omit contract_version, or pass the current version.",
                current_version=contract.version,
            )
        approval_ref = approval_ref or contract.approval_id
        self._check_approval(contract, approval_ref)
        if operation is not Operation.FIT or family is not Family.TABULAR:
            raise errors.invalid(
                "OPERATION_UNSUPPORTED",
                f"This version runs tabular 'fit' experiments only (got {family.value} "
                f"{operation.value}).",
                "Submit a baseline or candidate trial with operation 'fit'.",
            )
        if contract.family is not family:
            raise errors.invalid(
                "FAMILY_MISMATCH",
                f"The contract's family is {contract.family.value}.",
                "Omit family.",
            )
        limits = limits or contract.default_job_limits
        self._check_limits(limits, contract.default_job_limits)
        if not idempotency_key.startswith(f"{session.run_id}:"):
            raise errors.invalid(
                "IDEMPOTENCY_KEY_FORMAT",
                f"idempotency_key must start with this run's ID: '{session.run_id}:'.",
                f"Use a key such as '{session.run_id}:sub_0001'.",
            )
        try:
            request = SubmitExperimentRequest(
                run_id=session.run_id,
                trial_id=trial_id,
                operation=operation,
                family=family,
                contract_version=contract.version,
                model_ref=model_ref,
                config=config or {},
                limits=limits,
                approval_ref=approval_ref,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            raise errors.ToolFailure.from_error(errors.validation_failure(exc)) from None
        if (existing := self._existing_submission(request)) is not None:
            return existing
        trial = self._trial(trial_id)
        snapshot = self._trial_snapshot(trial_id)
        design = TrialFile.model_validate(yaml.safe_load(snapshot["trial.yaml"]))
        if design.operation is not operation:
            raise errors.invalid(
                "OPERATION_MISMATCH",
                f"trial.yaml declares operation {design.operation.value!r}.",
                "Rewrite the trial's design with operation 'fit'.",
            )
        dataset_id = session.config.experiment_dataset
        dataset_path = session.dataset_path(dataset_id)
        assert dataset_path is not None
        job_id = new_job_id()
        folder = protocol.job_dir(session.runs_root, session.run_id, job_id)
        (folder / protocol.TRIAL_SNAPSHOT_DIR).mkdir(parents=True)
        for name, content in snapshot.items():
            protocol.write_once(folder / protocol.TRIAL_SNAPSHOT_DIR / name, content)
        protocol.write_json_once(
            folder / protocol.REQUEST_FILE,
            protocol.ExperimentRequest(
                request=request,
                dataset_id=dataset_id,
                dataset_path=str(dataset_path),
                dataset_sha256=_sha256(dataset_path.read_bytes()),
                trial_files={name: _sha256(content) for name, content in snapshot.items()},
            ),
        )
        queued = session.queue.enqueue(
            trial_id=trial.trial_id,
            operation=operation,
            limits=limits,
            idempotency_key=idempotency_key,
            approval_ref=approval_ref,
            initiator=f"mcp:{session.config.session_id}",
            actor_role=session.role,
            job_id=job_id,
        )
        if queued != job_id:  # a concurrent call with the same key won the race
            return self._submitted(session.queue.get(queued), request, deduplicated=True)
        set_trial_status(session.database, trial_id, "submitted")
        return self._submitted(session.queue.get(job_id), request, deduplicated=False)

    def _check_approval(self, contract: ExperimentContract, approval_ref: str) -> None:
        recovery = "Omit approval_ref; experiments run under the contract's own approval."
        if approval_ref != contract.approval_id:
            raise errors.policy(
                "APPROVAL_MISMATCH",
                f"{approval_ref!r} is not the approval of contract v{contract.version}.",
                recovery,
                decided_by=PolicyRule.APPROVAL_STATE,
            )
        try:
            approval = load_approval(self.session.database, approval_ref)
        except RecordNotFound:
            approval = None
        if (
            approval is None
            or approval.run_id != self.session.run_id
            or approval.decision is not ApprovalDecision.APPROVED
            or approval.artifact_hash != contract.approved_plan_hash
        ):
            raise errors.policy(
                "APPROVAL_INVALID",
                "The contract's plan has no valid approval on record.",
                "Wait for the user to approve the plan.",
                decided_by=PolicyRule.APPROVAL_STATE,
            )

    @staticmethod
    def _check_limits(limits: JobLimits, ceiling: JobLimits) -> None:
        over = []
        if limits.max_wall_seconds > ceiling.max_wall_seconds:
            over.append(f"max_wall_seconds > {ceiling.max_wall_seconds}")
        if limits.max_memory_gb > ceiling.max_memory_gb:
            over.append(f"max_memory_gb > {ceiling.max_memory_gb}")
        if limits.gpu and not ceiling.gpu:
            over.append("gpu not approved")
        if (limits.cpu_threads or 1) > (ceiling.cpu_threads or 1):
            over.append(f"cpu_threads > {ceiling.cpu_threads or 1}")
        if over:
            raise errors.policy(
                "LIMITS_EXCEED_CONTRACT",
                f"Requested limits exceed the contract: {', '.join(over)}.",
                "Omit limits, or stay within the contract's defaults; only the user can extend "
                "them.",
                decided_by=PolicyRule.CONTRACT_SCOPE,
                ceiling=ceiling.model_dump(mode="json"),
            )

    def _existing_submission(self, request: SubmitExperimentRequest) -> SubmitResult | None:
        """Idempotency (spec §11): the same key and request returns the original job."""
        with self.session.database.engine.connect() as connection:
            job_id = connection.scalar(
                select(Job.id).where(Job.idempotency_key == request.idempotency_key)
            )
        if job_id is None:
            return None
        job = self.session.queue.get(job_id)
        folder = protocol.job_dir(self.session.runs_root, job.run_id, job.id)
        try:
            original = protocol.read_request(folder).request
        except FileNotFoundError:
            original = None
        if job.run_id != self.session.run_id or original != request:
            raise errors.invalid(
                "IDEMPOTENCY_KEY_REUSED",
                "This idempotency_key was already used for a different submission.",
                f"Use a new key such as '{self.session.run_id}:<new unique id>'.",
            )
        return self._submitted(job, request, deduplicated=True)

    def _trial_snapshot(self, trial_id: str) -> dict[str, bytes]:
        directory = self._trial_dir(trial_id)
        present = existing_files(
            self.session.runs_root, self._trial_dir_parts(trial_id), TRIAL_FILES
        )
        missing = sorted({"trial.yaml", "pipeline.py"} - set(present))
        if missing:
            raise errors.missing(
                "TRIAL_FILES_MISSING",
                f"Trial {trial_id!r} has no {missing}.",
                "Write the missing files with write_trial_files.",
            )
        return {
            name: read_output_bytes(directory, name, 2 * MAX_PIPELINE_BYTES) for name in present
        }

    def _submitted(
        self, job: JobRecord, request: SubmitExperimentRequest, *, deduplicated: bool
    ) -> SubmitResult:
        return SubmitResult(
            job_id=job.id,
            trial_id=job.trial_id,
            state=job.status,
            deduplicated=deduplicated,
            operation=job.operation,
            contract_version=request.contract_version,
            limits=job.limits,
            next_step=f"Poll get_experiment_status(job_id={job.id!r}) every few seconds.",
        )

    # --- get_experiment_status --------------------------------------------------------------------

    def get_experiment_status(
        self,
        job_id: Annotated[
            Identifier,
            Field(description="From submit_experiment, e.g. 'job_1761400000000000000_a1b2c3'."),
        ],
    ) -> ExperimentStatus:
        """Report where a submitted experiment is: queued, running, completed, or failed.

        Use it to poll after submit_experiment (every few seconds) until `finished` is true;
        then call get_experiment_metrics (if succeeded) or get_experiment_logs (if failed).

        Do not use it for metrics (get_experiment_metrics) or full logs (get_experiment_logs).

        Returns the job state, attempts, a platform-written error (code and message) if it
        failed, the last log lines (lines that showed dataset values are withheld), the IDs of
        stored artifacts, and the suggested next step.

        Example: get_experiment_status(job_id="job_1761400000000000000_a1b2c3")
        """
        job = self._job(job_id)
        folder = protocol.job_dir(self.session.runs_root, job.run_id, job.id)
        outcome = protocol.read_outcome(folder)
        finished = job.status in TERMINAL_STATES
        tail = _flatten(protocol.read_stage_logs(folder))[-20:]
        texts, withheld = self._screen([line.text for line in tail])
        error = None
        if outcome is not None and not outcome.succeeded:
            error = ExperimentError(
                code=outcome.error_code or "FAILED", message=(outcome.error_message or "")[:400]
            )
        succeeded = (job.status is JobState.COMPLETED) if finished else None
        return ExperimentStatus(
            job_id=job.id,
            trial_id=job.trial_id,
            operation=job.operation,
            state=job.status,
            attempts=job.attempts,
            finished=finished,
            succeeded=succeeded,
            error=error,
            log_tail=texts,
            withheld_log_lines=withheld,
            artifacts=[
                ArtifactRef.model_validate(artifact.model_dump())
                for artifact in (outcome.artifacts if outcome else [])
            ],
            next_step=_next_step(job, finished, succeeded),
        )

    # --- get_experiment_metrics -------------------------------------------------------------------

    def get_experiment_metrics(
        self,
        job_id: Annotated[
            Identifier,
            Field(description="A completed job from submit_experiment."),
        ],
    ) -> ExperimentMetrics:
        """Return the platform-computed validation scores of a completed experiment.

        Every number is computed by platform code from the trial's saved predictions, never by
        the trial itself. Use it once get_experiment_status reports `succeeded: true`, to compare
        trials on the contract's primary metric and decide what to try next.

        Do not use it before the job finishes (you get EXPERIMENT_NOT_FINISHED) or for test-set
        scores: the held-out test split is scored only for a frozen final candidate.

        Returns the primary metric and its direction, a mean/std/min/max summary of every metric
        across folds, per-fold scores (with confusion matrices for classification), per-class
        scores, and per-group spread for grouped splits (groups are unnamed: their labels are
        data values).

        Example: get_experiment_metrics(job_id="job_1761400000000000000_a1b2c3")
        """
        job = self._job(job_id)
        if job.status is not JobState.COMPLETED:
            if job.status in TERMINAL_STATES:
                raise errors.missing(
                    "EXPERIMENT_FAILED",
                    f"Job {job_id} ended {job.status.value}; it has no metrics.",
                    "Read get_experiment_status and get_experiment_logs, fix the trial, and "
                    "submit it again with a new idempotency_key.",
                )
            raise errors.not_yet(
                "EXPERIMENT_NOT_FINISHED",
                f"Job {job_id} is {job.status.value}.",
                "Poll get_experiment_status until finished, then call this again.",
                state=job.status.value,
            )
        folder = protocol.job_dir(self.session.runs_root, job.run_id, job.id)
        version = protocol.read_request(folder).request.contract_version
        contract = load_experiment_contract(self.session.database, job.run_id, version)
        stored = ArtifactStore(
            self.session.database, self.session.runs_root / job.run_id / "artifacts"
        ).get(f"{job.id}.metrics")  # hash-verified against the artifacts table
        scores = json.loads(stored.path.read_bytes())
        return _metrics(job, contract.primary_metric, scores)

    # --- get_experiment_logs ----------------------------------------------------------------------

    def get_experiment_logs(
        self,
        job_id: Annotated[Identifier, Field(description="From submit_experiment.")],
        page: Annotated[int, Field(ge=1, description="1-based page number.")] = 1,
        page_size: Annotated[int, Field(ge=1, le=200, description="Lines per page.")] = 50,
    ) -> ExperimentLogs:
        """Read a submitted experiment's sandbox output (stdout and stderr), page by page.

        Each sandbox stage (fit or predict, per validation fold) has its own log; lines are
        numbered within their stage. Use it to diagnose a failed or surprising trial: Python
        tracebacks, warnings, and training progress printed by pipeline.py.

        Do not use it to read data: every line that shows dataset values is replaced by a
        "withheld by the data firewall" marker, so printing rows from pipeline.py only hides
        your own log lines.

        Returns one page of lines (oldest first), the total line count, how many lines on this
        page were withheld, and `next_page` (null on the last page).

        Example: get_experiment_logs(job_id="job_1761400000000000000_a1b2c3", page=1)
        """
        job = self._job(job_id)
        folder = protocol.job_dir(self.session.runs_root, job.run_id, job.id)
        lines = _flatten(protocol.read_stage_logs(folder))
        start = (page - 1) * page_size
        chunk = lines[start : start + page_size]
        texts, withheld = self._screen([line.text for line in chunk])
        return ExperimentLogs(
            job_id=job.id,
            page=page,
            page_size=page_size,
            total_lines=len(lines),
            lines=[
                line.model_copy(update={"text": text})
                for line, text in zip(chunk, texts, strict=True)
            ],
            withheld_lines=withheld,
            next_page=page + 1 if start + page_size < len(lines) else None,
        )


# --- helpers ------------------------------------------------------------------------------------


def _flatten(stages: Sequence[protocol.StageLog]) -> list[LogLine]:
    return [
        LogLine(attempt=stage.attempt, stage=stage.stage, action=stage.action, line=n, text=text)
        for stage in stages
        for n, text in enumerate(stage.lines)
    ]


def _next_step(job: JobRecord, finished: bool, succeeded: bool | None) -> str:
    if not finished:
        return f"Still {job.status.value}: call get_experiment_status again in a few seconds."
    if succeeded:
        return f"Call get_experiment_metrics(job_id={job.id!r})."
    return (
        f"Read get_experiment_logs(job_id={job.id!r}), fix the trial with write_trial_files, "
        "then submit it again with a new idempotency_key."
    )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _summary(values: Sequence[float]) -> MetricSummary:
    return MetricSummary(
        mean=statistics.fmean(values),
        std=statistics.pstdev(values),
        min=min(values),
        max=max(values),
        folds=len(values),
    )


def _label(value: Any) -> str:
    return str(value)[:40]


def _metrics(job: JobRecord, primary: MetricName, scores: dict[str, Any]) -> ExperimentMetrics:
    """Shape the harness's metrics.json for an LLM: bounded, unnamed groups, cut labels."""
    folds_raw = scores.get("folds", [])
    folds: list[FoldScore] = []
    per_metric: dict[MetricName, list[float]] = {}
    per_class: dict[str, dict[str, list[float]]] = {}
    support: dict[str, int] = {}
    labels: list[str] = []
    known = {metric.value for metric in MetricName}
    for number, fold in enumerate(folds_raw):
        values = fold.get("metrics", {})
        metrics: dict[MetricName, float | None] = {}
        for name, value in values.items():
            if name in known:
                metric = MetricName(name)
                metrics[metric] = _finite(value)
                if metrics[metric] is not None:
                    per_metric.setdefault(metric, []).append(metrics[metric])  # type: ignore[arg-type]
        for entry in values.get("per_class", []):
            label = _label(entry["label"])
            if label not in labels:
                labels.append(label)
            bucket = per_class.setdefault(label, {"precision": [], "recall": [], "f1": []})
            for key in bucket:
                bucket[key].append(float(entry[key]))
            support[label] = support.get(label, 0) + int(entry["support"])
        matrix = values.get("confusion_matrix")
        groups = None
        if "per_group" in fold:
            group_scores = [
                score
                for group in fold["per_group"].values()
                if (score := _finite(group.get("metrics", {}).get(primary.value))) is not None
            ]
            ordered = sorted(group_scores)
            minimize = primary.direction.value == "minimize"
            groups = GroupSummary(
                n_groups=len(fold["per_group"]),
                mean=statistics.fmean(group_scores) if group_scores else None,
                worst=(ordered[-1] if minimize else ordered[0]) if ordered else None,
                best=(ordered[0] if minimize else ordered[-1]) if ordered else None,
            )
        folds.append(
            FoldScore(
                fold=number,
                rows=int(fold.get("rows", 0)),
                metrics=metrics,
                unavailable={
                    MetricName(name): str(reason)[:400]
                    for name, reason in fold.get("unavailable", {}).items()
                    if name in known
                },
                confusion_matrix=matrix if matrix is not None and len(matrix) <= 50 else None,
                groups=groups,
            )
        )
    shown = folds[:20]
    note = "Validation folds of the contract's split, scored by platform code."
    if len(folds) > len(shown):
        note += f" Showing 20 of {len(folds)} folds; the summary covers all of them."
    return ExperimentMetrics(
        job_id=job.id,
        trial_id=job.trial_id,
        split="validation",
        primary_metric=primary,
        direction=primary.direction,
        primary=_summary(per_metric[primary]) if per_metric.get(primary) else None,
        summary={metric: _summary(values) for metric, values in per_metric.items()},
        folds=shown,
        class_labels=labels[:50] if labels else None,
        per_class=[
            ClassScore(
                label=label,
                precision=statistics.fmean(per_class[label]["precision"]),
                recall=statistics.fmean(per_class[label]["recall"]),
                f1=statistics.fmean(per_class[label]["f1"]),
                support=support[label],
            )
            for label in labels[:50]
        ],
        note=note,
    )
