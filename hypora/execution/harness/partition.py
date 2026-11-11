"""Persist and verify deterministic, run-bound diagnostic and experiment partitions."""

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.core.schemas import ExperimentContract
from hypora.execution.harness.data import PreparedData, prepare_data
from hypora.execution.protocol.experiments import write_once

LOCK_FILE = "partition-lock.json"
_MAX_LOCK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class LockedPartition:
    """Verified prepared rows and the subset reserved for supervised diagnostics."""

    prepared: PreparedData
    diagnostic: np.ndarray


def _content(
    frame: pd.DataFrame, contract: ExperimentContract, dataset_sha256: str
) -> tuple[bytes, LockedPartition]:
    if len(dataset_sha256) != 64 or any(c not in "0123456789abcdef" for c in dataset_sha256):
        raise ValueError("invalid dataset fingerprint")
    try:
        prepared = prepare_data(frame, contract)
    except Exception:
        # Parser and sklearn errors can quote source values; never expose them to logs.
        raise ValueError("partition could not be prepared") from None
    data = contract.data_contract
    record = {
        "format": 1,
        "dataset_sha256": dataset_sha256,
        "run_id": contract.run_id,
        "target": contract.target,
        "task_type": contract.task_type.value,
        "seed": contract.seed,
        "split": data.split.model_dump(mode="json"),
        "non_learned_fixes": [
            item.model_dump(mode="json", by_alias=True) for item in data.non_learned_fixes
        ],
        "development": prepared.development.tolist(),
        "test": prepared.test.tolist(),
        "diagnostic": prepared.diagnostic.tolist(),
        "folds": [
            {"train": fold.train.tolist(), "validation": fold.validation.tolist()}
            for fold in prepared.folds
        ],
    }
    content = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(content) > _MAX_LOCK_BYTES:
        raise ValueError("partition lock exceeds the supported size")
    return content, LockedPartition(prepared, prepared.diagnostic)


def _read_lock(path: Path) -> bytes:
    # Do not follow a substituted symbolic link or read a non-regular or hard-linked file.
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > _MAX_LOCK_BYTES:
            raise ValueError("invalid partition lock file")
        return stream.read(_MAX_LOCK_BYTES + 1)


def create_partition_lock(
    frame: pd.DataFrame, contract: ExperimentContract, dataset_sha256: str, path: Path
) -> LockedPartition:
    """Create the lock exactly once, or verify an identical existing lock."""
    content, locked = _content(frame, contract, dataset_sha256)
    try:
        write_once(path, content)
    except FileExistsError:
        if _read_lock(path) != content:
            raise ValueError(
                "existing partition lock does not match the dataset and contract"
            ) from None
    return locked


def load_verified_partition(
    frame: pd.DataFrame, contract: ExperimentContract, dataset_sha256: str, path: Path
) -> LockedPartition:
    """Refuse missing or changed locks, then return the verified deterministic row IDs."""
    expected, locked = _content(frame, contract, dataset_sha256)
    actual = _read_lock(path)
    if not hashlib.sha256(actual).digest() == hashlib.sha256(expected).digest():
        raise ValueError("partition lock does not match the dataset and contract")
    return locked
