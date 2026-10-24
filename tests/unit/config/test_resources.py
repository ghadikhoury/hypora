"""Hardware detection (works without a GPU) and pool resolution (spec §9.2)."""

import shutil
import subprocess
from typing import Any

import pytest

from hypora.core.config.resources import (
    DetectedResources,
    GpuDetection,
    GpuInfo,
    ResourceConfigError,
    ResourcesConfig,
    detect_gpus,
    detect_resources,
    resolve_pools,
)

TWO_GPUS = "0, NVIDIA GeForce RTX 3060, 12288\n1, NVIDIA A100-SXM4-40GB, 40960\n"


def fake_runner(stdout: str = "", returncode: int = 0, raises: BaseException | None = None) -> Any:
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert kwargs["timeout"] > 0, "nvidia-smi must always be called with a timeout"
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="boom")

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.fixture
def nvidia_smi_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


def test_machine_without_nvidia_smi_reports_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    runner = fake_runner()
    gpus, status, _ = detect_gpus(5.0, runner)
    assert (gpus, status) == ((), GpuDetection.NOT_FOUND)
    assert runner.calls == []  # nothing is executed


@pytest.mark.usefixtures("nvidia_smi_on_path")
def test_gpus_are_parsed_from_nvidia_smi() -> None:
    runner = fake_runner(TWO_GPUS)
    gpus, status, _ = detect_gpus(5.0, runner)
    assert status is GpuDetection.OK
    assert gpus == (
        GpuInfo(index=0, name="NVIDIA GeForce RTX 3060", memory_gb=12.0),
        GpuInfo(index=1, name="NVIDIA A100-SXM4-40GB", memory_gb=40.0),
    )
    assert runner.calls[0][0] == "nvidia-smi"


@pytest.mark.usefixtures("nvidia_smi_on_path")
@pytest.mark.parametrize(
    ("runner", "status"),
    [
        (fake_runner(raises=subprocess.TimeoutExpired("nvidia-smi", 5.0)), GpuDetection.TIMEOUT),
        (fake_runner(returncode=9), GpuDetection.ERROR),
        (fake_runner(raises=PermissionError("denied")), GpuDetection.ERROR),
        (fake_runner("garbage without commas\n"), GpuDetection.ERROR),
    ],
    ids=["timeout", "nonzero_exit", "cannot_execute", "unparseable"],
)
def test_gpu_detection_failures_mean_no_gpus_not_a_crash(runner: Any, status: GpuDetection) -> None:
    gpus, got, detail = detect_gpus(5.0, runner)
    assert (gpus, got) == ((), status)
    assert detail


def test_detect_resources_on_this_machine_never_raises() -> None:
    detected = detect_resources(timeout_s=10.0)
    assert detected.cpu_cores >= 1
    assert detected.gpu_detection in GpuDetection
    assert len(detected.gpus) == 0 or detected.gpu_detection is GpuDetection.OK


def test_detection_does_not_import_ml_frameworks() -> None:
    import sys

    detect_resources(timeout_s=10.0)
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules


# --- Pools ----------------------------------------------------------------------------------------

CPU_ONLY = DetectedResources(
    cpu_cores=8, total_memory_gb=16.0, gpu_detection=GpuDetection.NOT_FOUND
)
ONE_GPU = DetectedResources(
    cpu_cores=8,
    total_memory_gb=16.0,
    gpus=(GpuInfo(index=0, name="RTX", memory_gb=12.0),),
    gpu_detection=GpuDetection.OK,
)


def test_default_pools_follow_spec_9_2() -> None:
    pools = resolve_pools(ONE_GPU, ResourcesConfig(memory_reserve_gb=2.0))
    assert (pools.cpu_slots, pools.gpu_slots, pools.memory_budget_gb) == (7, 1, 14.0)


def test_pools_on_a_machine_without_gpu() -> None:
    pools = resolve_pools(CPU_ONLY, ResourcesConfig())
    assert pools.gpu_slots == 0
    assert pools.cpu_slots == 7


def test_single_core_machine_still_gets_a_slot() -> None:
    one_core = CPU_ONLY.model_copy(update={"cpu_cores": 1})
    assert resolve_pools(one_core, ResourcesConfig()).cpu_slots == 1


def test_overrides_may_shrink_pools() -> None:
    config = ResourcesConfig(cpu_slots=2, gpu_slots=0, memory_budget_gb=4.0)
    pools = resolve_pools(ONE_GPU, config)
    assert (pools.cpu_slots, pools.gpu_slots, pools.memory_budget_gb) == (2, 0, 4.0)


@pytest.mark.parametrize(
    ("detected", "config", "match"),
    [
        (CPU_ONLY, ResourcesConfig(gpu_slots=1), "never oversubscribed"),
        (CPU_ONLY, ResourcesConfig(cpu_slots=9), "exceeds the 8 detected cores"),
        (CPU_ONLY, ResourcesConfig(memory_reserve_gb=16.0), "leaves no RAM"),
        (
            CPU_ONLY.model_copy(update={"total_memory_gb": None}),
            ResourcesConfig(),
            "set resources.memory_budget_gb",
        ),
    ],
)
def test_impossible_pools_fail_fast(
    detected: DetectedResources, config: ResourcesConfig, match: str
) -> None:
    with pytest.raises(ResourceConfigError, match=match):
        resolve_pools(detected, config)


@pytest.mark.parametrize("memory", [1000.0, 16.0, 14.00000001])
def test_explicit_memory_cannot_exceed_ram_minus_reserve(memory: float) -> None:
    with pytest.raises(ResourceConfigError, match=r"memory_budget_gb.*exceeds"):
        resolve_pools(CPU_ONLY, ResourcesConfig(memory_budget_gb=memory, memory_reserve_gb=2))


@pytest.mark.parametrize("reserve", [16.0, 17.0])
def test_explicit_memory_does_not_bypass_impossible_reserve(reserve: float) -> None:
    with pytest.raises(ResourceConfigError, match="leaves no RAM"):
        resolve_pools(CPU_ONLY, ResourcesConfig(memory_budget_gb=1, memory_reserve_gb=reserve))


@pytest.mark.parametrize("memory", [4.0, 14.0])
def test_valid_memory_override_including_exact_boundary(memory: float) -> None:
    pools = resolve_pools(CPU_ONLY, ResourcesConfig(memory_budget_gb=memory, memory_reserve_gb=2))
    assert pools.memory_budget_gb == memory


def test_undetectable_ram_still_allows_manual_pool() -> None:
    unknown = CPU_ONLY.model_copy(update={"total_memory_gb": None})
    pools = resolve_pools(unknown, ResourcesConfig(memory_budget_gb=4))
    assert pools.memory_budget_gb == 4


def test_default_memory_pool_is_never_rounded_above_available_ram() -> None:
    detected = CPU_ONLY.model_copy(update={"total_memory_gb": 16.0006})
    assert detected.total_memory_gb is not None
    pools = resolve_pools(detected, ResourcesConfig(memory_reserve_gb=2))
    assert pools.memory_budget_gb == 16.0006 - 2
    assert pools.memory_budget_gb <= detected.total_memory_gb - 2


@pytest.mark.parametrize("memory", [13.9996, 0.0001])
def test_explicit_memory_is_preserved_without_rounding(memory: float) -> None:
    assert (
        resolve_pools(CPU_ONLY, ResourcesConfig(memory_budget_gb=memory)).memory_budget_gb == memory
    )
