"""Hardware detection and resource pools (spec §9.2).

Detection never raises and never imports an ML framework: CPU cores from the OS (respecting CPU
affinity), RAM from `sysconf`, GPUs by calling `nvidia-smi` with a timeout. A machine without a GPU
(or without `nvidia-smi`) simply reports zero GPUs.
"""

import os
import shutil
import subprocess
from collections.abc import Callable
from enum import StrEnum

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from hypora.core.schemas.common import FrozenModel, NonEmptyStr

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total",
    "--format=csv,noheader,nounits",
]


class LlmConcurrency(FrozenModel):
    """Concurrent calls allowed per provider (`llm_concurrency[provider]`, spec §9.2)."""

    gemini: PositiveInt = 2
    openai: PositiveInt = 4
    anthropic: PositiveInt = 2


class ResourcesConfig(FrozenModel):
    """`configs/resources.yaml`. `null` pool sizes mean "derive from detected hardware"."""

    cpu_slots: PositiveInt | None = None
    gpu_slots: NonNegativeInt | None = None
    memory_budget_gb: PositiveFloat | None = None
    memory_reserve_gb: NonNegativeFloat = Field(2.0, description="RAM kept for the OS and plane.")
    gpu_detection_timeout_s: PositiveFloat = 5.0
    llm_concurrency: LlmConcurrency = LlmConcurrency()


class GpuDetection(StrEnum):
    OK = "ok"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    ERROR = "error"


class GpuInfo(FrozenModel):
    index: NonNegativeInt
    name: NonEmptyStr
    memory_gb: NonNegativeFloat


class DetectedResources(FrozenModel):
    cpu_cores: PositiveInt
    total_memory_gb: NonNegativeFloat | None = Field(None, description="None if undetectable.")
    gpus: tuple[GpuInfo, ...] = ()
    gpu_detection: GpuDetection
    gpu_detection_detail: str = ""


class ResourcePools(FrozenModel):
    """Pool sizes the scheduler enforces (spec §9.2)."""

    cpu_slots: PositiveInt
    gpu_slots: NonNegativeInt
    memory_budget_gb: PositiveFloat
    llm_concurrency: LlmConcurrency


class ResourceConfigError(ValueError):
    """Configured pools are impossible on the detected hardware."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def detect_cpu_cores() -> int:
    """Cores this process may use (affinity-aware on Linux), at least 1."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def detect_total_memory_gb() -> float | None:
    """Physical RAM in GiB via `sysconf`, or None where unavailable."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return None


def detect_gpus(
    timeout_s: float, runner: Runner = subprocess.run
) -> tuple[tuple[GpuInfo, ...], GpuDetection, str]:
    """Query `nvidia-smi`. Returns (gpus, status, detail); never raises."""
    if shutil.which(NVIDIA_SMI_QUERY[0]) is None:
        return (), GpuDetection.NOT_FOUND, "nvidia-smi not on PATH"
    try:
        result = runner(NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return (), GpuDetection.TIMEOUT, f"nvidia-smi did not answer within {timeout_s} s"
    except OSError as exc:
        return (), GpuDetection.ERROR, f"nvidia-smi could not run: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:200]
        return (), GpuDetection.ERROR, f"nvidia-smi exited {result.returncode}: {detail}"
    try:
        return _parse_nvidia_smi(result.stdout), GpuDetection.OK, ""
    except ValueError as exc:
        return (), GpuDetection.ERROR, f"unparseable nvidia-smi output: {exc}"


def _parse_nvidia_smi(stdout: str) -> tuple[GpuInfo, ...]:
    gpus = []
    for line in stdout.strip().splitlines():
        index, name, memory_mib = (part.strip() for part in line.split(","))
        gpus.append(GpuInfo(index=int(index), name=name, memory_gb=float(memory_mib) / 1024))
    return tuple(gpus)


def detect_resources(timeout_s: float = 5.0, runner: Runner = subprocess.run) -> DetectedResources:
    gpus, status, detail = detect_gpus(timeout_s, runner)
    return DetectedResources(
        cpu_cores=detect_cpu_cores(),
        total_memory_gb=detect_total_memory_gb(),
        gpus=gpus,
        gpu_detection=status,
        gpu_detection_detail=detail,
    )


def resolve_pools(detected: DetectedResources, config: ResourcesConfig) -> ResourcePools:
    """Default pool sizes from spec §9.2, overridable in resources.yaml; never oversubscribe."""
    cpu_slots = config.cpu_slots or max(1, detected.cpu_cores - 1)
    if cpu_slots > detected.cpu_cores:
        raise ResourceConfigError(
            f"resources.cpu_slots={cpu_slots} exceeds the {detected.cpu_cores} detected cores"
        )

    gpu_slots = len(detected.gpus) if config.gpu_slots is None else config.gpu_slots
    if gpu_slots > len(detected.gpus):
        raise ResourceConfigError(
            f"resources.gpu_slots={gpu_slots} but {len(detected.gpus)} GPU(s) were detected "
            f"({detected.gpu_detection.value}); GPUs are never oversubscribed"
        )

    if detected.total_memory_gb is None:
        if config.memory_budget_gb is None:
            raise ResourceConfigError("RAM could not be detected: set resources.memory_budget_gb")
        # Only the undetectable-RAM case must rely on the operator's explicit capacity.
        memory = config.memory_budget_gb
    else:
        available = detected.total_memory_gb - config.memory_reserve_gb
        if available <= 0:
            raise ResourceConfigError(
                f"memory_reserve_gb={config.memory_reserve_gb} leaves no RAM of "
                f"{detected.total_memory_gb:.1f} GiB"
            )
        memory = available if config.memory_budget_gb is None else config.memory_budget_gb
        if memory > available:
            raise ResourceConfigError(
                f"resources.memory_budget_gb={memory} exceeds the {available} GiB available "
                f"after memory_reserve_gb={config.memory_reserve_gb} is subtracted from "
                f"{detected.total_memory_gb} GiB detected RAM"
            )
    return ResourcePools(
        cpu_slots=cpu_slots,
        gpu_slots=gpu_slots,
        # Rounding for display must not increase the scheduler's capacity or erase a small pool.
        memory_budget_gb=memory,
        llm_concurrency=config.llm_concurrency,
    )
