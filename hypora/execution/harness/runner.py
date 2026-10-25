"""Execution-plane harness. It never imports trial code or deserializes a model on the host."""

import hashlib
import io
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from hypora.core.schemas import ExperimentContract, JobLimits
from hypora.execution.harness.data import PreparedData, prepare_data
from hypora.execution.harness.metrics import score_predictions
from hypora.execution.harness.planner import HarnessPlanner, HarnessTask
from hypora.execution.sandbox.launcher import SandboxLauncher
from hypora.execution.sandbox.outputs import read_output_bytes, scan_outputs


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Encode a platform-owned input with the project's pinned Parquet engine."""
    return frame.reset_index(drop=True).to_parquet(engine="pyarrow", index=False)


def load_tabular(path: Path, *, source: bytes | None = None) -> pd.DataFrame:
    """Read an authorized worker-local CSV or Parquet dataset, never a model artifact."""
    if path.suffix.lower() not in (".csv", ".parquet"):
        raise ValueError("tabular input must be CSV or Parquet")
    source = path.read_bytes() if source is None else source
    if path.suffix.lower() == ".csv":
        return pd.read_csv(io.BytesIO(source))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(io.BytesIO(source), engine="pyarrow")
    raise ValueError("tabular input must be CSV or Parquet")


def digest(content: bytes) -> str:
    """Content-address a dataset, code snapshot, model, or prediction artifact."""
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class ValidationResult:
    """Validation evidence and worker-local references; not an authorization to read test labels."""

    prepared: PreparedData
    pipeline: bytes
    contract_hash: str
    trial_id: str
    dataset_ref: str
    dataset_fingerprint: str
    scores: dict
    config_json: bytes
    artifacts: Path
    resources: list[dict]

    @property
    def config(self) -> dict:
        """Return a fresh copy of the exact validation configuration."""
        return json.loads(self.config_json)


@dataclass(frozen=True)
class FrozenCandidate:
    """Final-fit model and exact code snapshot, frozen before the one-shot test score."""

    validation: ValidationResult
    model: bytes
    model_hash: str
    pipeline_hash: str
    freeze_id: str


class HarnessFailure(RuntimeError):
    """A sandbox failed or produced outputs that cannot be scored."""


class TabularHarness:
    """Run on the execution worker. Labels and model artifacts stay in this execution plane.

    validate -> freeze_candidate -> score_test is explicit: validation never touches test labels
    inside trial code, and final scoring requires the exact frozen model/code and contract.
    """

    def __init__(
        self,
        root: Path,
        image: str,
        launcher: SandboxLauncher,
        *,
        worker_id: str = "harness-worker",
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.launcher = launcher
        self.worker_id = worker_id
        self.planner = HarnessPlanner(self.root / "jobs", image)
        self.resources: list[dict] = []
        self._scored: set[str] = set()

    def _execute(self, task: HarnessTask) -> Path:
        job_id = "harness_" + uuid4().hex
        self.planner.register(job_id, task)
        spec = self.planner.build(job_id, self.worker_id)
        result = self.launcher.run(spec)
        self.resources.append(
            {
                "job_id": job_id,
                "action": task.action,
                "status": result.status.value,
                "wall_seconds": result.duration_s,
                "limits": task.limits.model_dump(mode="json"),
            }
        )
        if not result.ok:
            raise HarnessFailure(f"{result.status.value}: {result.detail}; {result.logs}")
        resource_report = json.loads(
            read_output_bytes(spec.mounts.out, "resources.json", 64 * 1024)
        )
        self.resources[-1]["process_report"] = resource_report
        return spec.mounts.out

    def _context(self, contract, config):
        return {"seed": contract.seed, "task_type": contract.task_type.value, "config": config}

    def _fit(self, data, ids, pipeline, contract, config, validation=None, limits=None):
        limits = limits or contract.default_job_limits
        inputs = {
            "train_features.parquet": parquet_bytes(data.features.iloc[ids]),
            "train_labels.parquet": parquet_bytes(pd.DataFrame({"target": data.target.iloc[ids]})),
        }
        if validation is not None:
            inputs["validation_features.parquet"] = parquet_bytes(data.features.iloc[validation])
            inputs["validation_labels.parquet"] = parquet_bytes(
                pd.DataFrame({"target": data.target.iloc[validation]})
            )
        out = self._execute(
            HarnessTask(
                "fit",
                pipeline,
                inputs,
                self._context(contract, config),
                limits,
            )
        )
        return read_output_bytes(out, "model.joblib", int(limits.max_memory_gb * 1024**3))

    def _predict(self, features, model, pipeline, contract, config, limits=None):
        # Even if a model's pickle executes code during load, this stage has features only.
        inputs = {"features.parquet": parquet_bytes(features), "model.joblib": model}
        out = self._execute(
            HarnessTask(
                "predict",
                pipeline,
                inputs,
                self._context(contract, config),
                limits or contract.default_job_limits,
            )
        )
        raw = read_output_bytes(out, "predictions.parquet", 512 * 1024**2)
        table = pq.ParquetFile(io.BytesIO(raw))
        metadata = table.metadata
        uncompressed = sum(
            metadata.row_group(i).column(j).total_uncompressed_size
            for i in range(metadata.num_row_groups)
            for j in range(metadata.num_columns)
        )
        if (
            metadata.num_rows != len(features)
            or metadata.num_columns > 128
            or uncompressed > 512 * 1024**2
        ):
            raise HarnessFailure("prediction Parquet exceeds expected rows/columns/decoded size")
        predictions = table.read().to_pandas()
        classes = json.loads(read_output_bytes(out, "classes.json", 64 * 1024))
        if not isinstance(classes, list) or len(classes) > 127:
            raise HarnessFailure("invalid prediction class metadata")
        return predictions, classes

    def _score(self, data, ids, predictions, classes, contract, train_ids):
        if contract.task_type.value != "regression":
            expected = sorted(data.target.iloc[train_ids].unique().tolist())
            if classes != expected:
                raise HarnessFailure("prediction class order differs from training labels")
        key = contract.data_contract.split.group_key
        groups = data.frame.iloc[ids][key].to_numpy() if key else None
        scores = score_predictions(
            data.target.iloc[ids], predictions, contract.task_type.value, classes, groups
        )
        required = [contract.primary_metric, *contract.secondary_metrics]
        for metric in required:
            if metric.value not in scores["metrics"]:
                raise HarnessFailure(f"contract metric unsupported for task: {metric.value}")
        return scores

    def validate(
        self,
        frame: pd.DataFrame | Path,
        contract: ExperimentContract,
        trial_dir: Path,
        *,
        trial_id: str,
        dataset_ref: str,
        config: dict | None = None,
        limits: JobLimits | None = None,
    ) -> ValidationResult:
        """Fit every validation fold and recompute scores from saved predictions on the worker.

        `limits` are the submitted job's own sandbox limits, applied to every stage; by default
        the contract's `default_job_limits`. The caller checks they stay within the contract.
        """
        if isinstance(frame, Path):
            source = frame.read_bytes()
            fingerprint = digest(source)
            frame = load_tabular(frame, source=source)
        else:
            fingerprint = digest(parquet_bytes(frame))
        resource_start = len(self.resources)
        data = prepare_data(frame, contract)
        pipeline = read_output_bytes(trial_dir.resolve(), "pipeline.py", 2 * 1024**2)
        artifact_dir = self.root / "artifacts" / uuid4().hex
        artifact_dir.mkdir(parents=True)
        all_predictions = []
        fold_scores = []
        fold_classes = []
        for number, fold in enumerate(data.folds):
            model = self._fit(
                data, fold.train, pipeline, contract, config or {}, fold.validation, limits
            )
            predictions, classes = self._predict(
                data.features.iloc[fold.validation], model, pipeline, contract, config or {}, limits
            )
            fold_classes.append(classes)
            predictions.insert(0, "row_id", fold.validation)
            predictions.insert(0, "fold", number)
            all_predictions.append(predictions)
        self._write(
            artifact_dir / "predictions.parquet",
            parquet_bytes(pd.concat(all_predictions, ignore_index=True)),
        )
        saved = pd.read_parquet(
            io.BytesIO(read_output_bytes(artifact_dir, "predictions.parquet", 512 * 1024**2)),
            engine="pyarrow",
        )
        for number, fold in enumerate(data.folds):
            predictions = saved.loc[saved["fold"] == number].drop(columns=["fold", "row_id"])
            fold_scores.append(
                self._score(
                    data, fold.validation, predictions, fold_classes[number], contract, fold.train
                )
            )
        scores = {"folds": fold_scores}
        self._write(artifact_dir / "metrics.json", self._json(scores))
        result = ValidationResult(
            data,
            pipeline,
            contract.hash,
            trial_id,
            dataset_ref,
            fingerprint,
            scores,
            self._json(config or {}),
            artifact_dir,
            self.resources[resource_start:],
        )
        self._manifest(result, contract, config or {}, "validation")
        return result

    def freeze_candidate(
        self,
        validation: ValidationResult,
        contract: ExperimentContract,
        *,
        config: dict | None = None,
    ) -> FrozenCandidate:
        """Refit on development only; freeze the candidate before any test scoring."""
        self._check_contract(validation, contract)
        if config is not None and config != validation.config:
            raise HarnessFailure("configuration changed after validation")
        try:
            with (validation.artifacts / "freeze-started").open("xb"):
                pass
        except FileExistsError as exc:
            raise HarnessFailure("candidate already frozen or final fit already attempted") from exc
        resource_start = len(self.resources)
        model = self._fit(
            validation.prepared,
            validation.prepared.development,
            validation.pipeline,
            contract,
            validation.config,
        )
        validation.resources.extend(self.resources[resource_start:])
        frozen = FrozenCandidate(
            validation, model, digest(model), digest(validation.pipeline), uuid4().hex
        )
        self._write(validation.artifacts / "model.joblib", model)
        self._write(
            validation.artifacts / "frozen.json",
            self._json(
                {
                    "freeze_id": frozen.freeze_id,
                    "model_sha256": frozen.model_hash,
                    "pipeline_sha256": frozen.pipeline_hash,
                    "contract_hash": contract.hash,
                }
            ),
        )
        return frozen

    def score_test(
        self,
        candidate: FrozenCandidate,
        contract: ExperimentContract,
        *,
        config: dict | None = None,
    ) -> dict:
        """Score one frozen final candidate once; the container receives test features only."""
        validation = candidate.validation
        self._check_contract(validation, contract)
        if (
            digest(candidate.model) != candidate.model_hash
            or digest(validation.pipeline) != candidate.pipeline_hash
        ):
            raise HarnessFailure("frozen model or pipeline changed")
        if config is not None and config != validation.config:
            raise HarnessFailure("configuration changed after freezing")
        marker = validation.artifacts / ("test-started-" + candidate.freeze_id)
        if candidate.freeze_id in self._scored:
            raise HarnessFailure("candidate has already consumed its test score")
        # Durable one-shot marker: even a failed test attempt cannot become test-set tuning.
        try:
            with marker.open("xb"):
                pass
        except FileExistsError as exc:
            raise HarnessFailure("candidate has already consumed its test score") from exc
        self._scored.add(candidate.freeze_id)
        data, ids = validation.prepared, validation.prepared.test
        resource_start = len(self.resources)
        predictions, classes = self._predict(
            data.features.iloc[ids],
            candidate.model,
            validation.pipeline,
            contract,
            validation.config,
        )
        validation.resources.extend(self.resources[resource_start:])
        self._write(validation.artifacts / "test-predictions.parquet", parquet_bytes(predictions))
        # Platform reads its saved prediction file, not any metric reported by trial code.
        predictions = pd.read_parquet(
            validation.artifacts / "test-predictions.parquet", engine="pyarrow"
        )
        scores = self._score(data, ids, predictions, classes, contract, data.development)
        self._write(validation.artifacts / "test-metrics.json", self._json(scores))
        self._manifest(validation, contract, validation.config, "test", candidate)
        return scores

    def _check_contract(self, validation, contract):
        if validation.contract_hash != contract.hash:
            raise HarnessFailure("contract changed after validation")

    def _manifest(self, validation, contract, config, phase, candidate=None):
        files = scan_outputs(validation.artifacts)
        if not files.safe:
            raise HarnessFailure("unsafe artifact directory")
        data = validation.prepared
        manifest = {
            "trial_id": validation.trial_id,
            "family": "tabular",
            "phase": phase,
            "contract_version": contract.version,
            "contract_hash": contract.hash,
            "dataset_ref": validation.dataset_ref,
            "dataset_fingerprint": validation.dataset_fingerprint,
            "split": contract.data_contract.split.model_dump(mode="json"),
            "row_ids": {
                "development": data.development.tolist(),
                "test": data.test.tolist(),
                "folds": [
                    {"train": f.train.tolist(), "validation": f.validation.tolist()}
                    for f in data.folds
                ],
            },
            "allowed_features": list(contract.data_contract.allowed_features),
            "model_source": "trial:pipeline.py",
            "model_revision": digest(validation.pipeline),
            "seed": contract.seed,
            "config": config,
            "worker_id": self.worker_id,
            "worker_python_version": platform.python_version(),
            "python_version": validation.resources[-1]["process_report"]["python_version"],
            "sandbox_packages": validation.resources[-1]["process_report"]["packages"],
            "image": self.planner.image,
            "image_digest": self.launcher.client.images.get(self.planner.image).id,
            "lockfile_sha256": digest(
                read_output_bytes(
                    Path(__file__).parents[1] / "sandbox", "requirements-tabular.lock", 4 * 1024**2
                )
            ),
            "resources": validation.resources,
            "metrics": validation.scores,
            "artifacts": [
                {"path": f.path, "sha256": f.sha256, "size": f.size}
                for f in files.files
                if f.path != "manifest.yaml"
            ],
        }
        if candidate:
            manifest["frozen_model_sha256"] = candidate.model_hash
            manifest["test_metrics"] = json.loads(
                read_output_bytes(validation.artifacts, "test-metrics.json", 4 * 1024**2)
            )
        self._write(
            validation.artifacts / "manifest.yaml",
            yaml.safe_dump(manifest, sort_keys=False).encode(),
        )

    @staticmethod
    def _json(value):
        return json.dumps(value, indent=2, allow_nan=False).encode()

    @staticmethod
    def _write(path, content):
        # These directories are platform-owned and never mounted into a sandbox.
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
