"""Resource-pool accounting for one worker (spec §9.2): CPU slots, GPU slots, memory budget."""

import threading
from dataclasses import dataclass

from hypora.core.config.resources import ResourcePools
from hypora.core.schemas import JobLimits
from hypora.storage.jobs import WorkerCapacity


@dataclass(frozen=True)
class JobNeeds:
    cpu_slots: int
    gpu_slots: int
    memory_gb: float

    @classmethod
    def of(cls, limits: JobLimits) -> "JobNeeds":
        """A job needs `cpu_threads` CPU slots (1 if unset), one GPU if it asks, its memory."""
        return cls(limits.cpu_threads or 1, 1 if limits.gpu else 0, limits.max_memory_gb)


class PoolAccountant:
    """Free capacity across the jobs this worker is running. Thread-safe; never oversubscribes."""

    def __init__(self, pools: ResourcePools) -> None:
        self.pools = pools
        self._lock = threading.Lock()
        self._cpu = 0
        self._gpu = 0
        self._memory = 0.0

    def free(self) -> WorkerCapacity:
        with self._lock:
            return WorkerCapacity(
                cpu_slots=self.pools.cpu_slots - self._cpu,
                gpu_slots=self.pools.gpu_slots - self._gpu,
                memory_gb=self.pools.memory_budget_gb - self._memory,
            )

    def reserve(self, limits: JobLimits) -> bool:
        """Claim a job's resources if they fit; False (and nothing claimed) otherwise."""
        needs = JobNeeds.of(limits)
        with self._lock:
            fits = (
                self._cpu + needs.cpu_slots <= self.pools.cpu_slots
                and self._gpu + needs.gpu_slots <= self.pools.gpu_slots
                and self._memory + needs.memory_gb <= self.pools.memory_budget_gb
            )
            if fits:
                self._cpu += needs.cpu_slots
                self._gpu += needs.gpu_slots
                self._memory += needs.memory_gb
            return fits

    def release(self, limits: JobLimits) -> None:
        needs = JobNeeds.of(limits)
        with self._lock:
            self._cpu -= needs.cpu_slots
            self._gpu -= needs.gpu_slots
            self._memory -= needs.memory_gb
