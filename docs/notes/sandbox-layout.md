# Sandbox layout, isolation, and the job queue (S1.2)

Packages: `hypora/execution/sandbox/` (image, spec, launcher, outputs), `hypora/execution/worker/`
(worker loop, pools, planners), `hypora/storage/jobs.py` (job queue). Decisions: D-009.

## The fixed mount layout (S1.3 depends on this)

| Path in the container | Mode | Holds | Host side |
|---|---|---|---|
| `/data` | read-only | this job's input data only (S1.3 prepares the split it may see) | any directory |
| `/trial` | read-write | this trial's directory: `trial.yaml`, `pipeline.py`, ... | `runs/{run_id}/trials/{trial_id}/` |
| `/harness` | read-only | platform harness code | the harness package directory |
| `/out` | read-write, **fresh and empty per job** | results of this job only | a new directory per job |
| `/tmp` | tmpfs | scratch space, size-capped, `noexec,nosuid,nodev` | none (memory) |

Everything else is the read-only image. The working directory is `/trial`; `HOME`, `TMPDIR`, and
`MPLCONFIGDIR` point at `/tmp`.

**Before anything starts,** `SandboxMounts.validated()` resolves every host path (symlinks
included) and refuses the job if:
- a source is missing or not a directory;
- `/out` is not empty;
- any two sources are the same directory or nested in one another.

The nesting rule matters: data or harness inside the trial directory would become writable through
`/trial`, and an `/out` inside it would expose earlier results.

**"Write-only /out".** Docker bind mounts can only be read-only or read-write. So `/out` is a new
empty directory for every job (D-009, option 2a): trial code can only ever read back what the same
job wrote, and the platform reads it afterwards only through the output reader below.

## Reading job outputs: never follow a link (binding for S1.3 and every later reader)

Trial code controls everything in `/trial` and `/out`. A symlink it creates there, like
`model.bin -> /etc/passwd` or `metrics.json -> /home/<user>/.ssh/id_rsa`, becomes a **real host
symlink**. Measured on WSL ext4 and on the Windows drive alike: host code that opens it reads the
host file. So:

- After every job, and after the container has been removed (no process can still write), the
  worker scans `/trial` and `/out` with `lstat` semantics. Any symlink or special file (FIFO,
  socket, device) fails the job with `UNSAFE_OUTPUT`, and the entry is unlinked. The link itself is
  removed, never its target.
- Platform code reads outputs **only** through `hypora.execution.sandbox.outputs`:
  - `scan_outputs(root)` returns the regular files (path, size, SHA-256) and the rejected entries.
  - `open_output(root, relpath)` and `read_output_bytes(root, relpath, max_bytes)` open each path
    component through `O_NOFOLLOW` directory file descriptors. They accept only plain relative paths
    (no `/`, `..`, or `.`) and only regular files.

```python
from hypora.execution.sandbox.outputs import read_output_bytes, scan_outputs

manifest = scan_outputs(out_dir)              # already checked by the worker, safe to repeat
metrics = read_output_bytes(out_dir, "metrics.json", max_bytes=10 * 1024 * 1024)
```

Never `open()`, `Path.read_*()`, `shutil.copy*()`, or `pandas.read_*()` a path inside a job's
`/trial` or `/out` directly. The same holds for writes: the trial directory persists across a
trial's jobs, so S1.4's `write_trial_files` must not follow links when it writes there either.

## Isolation applied to every container (`launcher.py`)

| Control | Setting | Tested by (`tests/sandbox/test_security.py`) |
|---|---|---|
| User | the worker's own non-root UID:GID; UID 10001 if the worker is root; never 0 | `test_runs_as_the_workers_non_root_uid_without_capabilities` |
| Capabilities | `cap_drop ALL` (effective, permitted, and bounding sets are all empty) | same |
| Privilege escalation | `no-new-privileges`; not privileged; Docker's default seccomp profile | same |
| Root filesystem | read-only | `test_writes_outside_trial_and_out_fail` |
| `/tmp` | tmpfs, size-capped, `noexec` | `test_tmp_is_size_capped_and_not_executable` |
| Network | `none` (only `lo`): no TCP, DNS, or HTTP | `test_network_access_fails` |
| Host files | nothing but the four mounts; no Docker socket, no home directory | `test_non_mounted_host_paths_are_invisible` |
| Environment | only the platform defaults + the spec's `env`; the worker's environment never leaks | `test_worker_environment_never_leaks` |
| Memory | `mem_limit` = `memswap_limit` (no swap) → OOM kill | `test_memory_limit_kills_the_job` |
| Processes | `pids_limit` | `test_process_limit_contains_a_fork_bomb` |
| CPU | `nano_cpus`; thread variables match | — |
| One file's size | `RLIMIT_FSIZE` (write fails with EFBIG) | `test_single_file_size_is_capped` |
| Total output | `/trial` + `/out` size polled during the run and checked after it | `test_total_output_is_capped` |
| Wall clock | polled; kill at the limit | `test_timeout_kills_the_job` |
| Lost lease / cancel | polled `should_abort`; kill | `test_abort_request_kills_the_job` |
| Output links | no-symlink rule above | `test_symlink_escape_through_outputs_is_refused` |
| Unsafe specs | refused before start | `test_overlapping_mounts_are_refused` and 2 more |
| Orphans | labelled containers; `cleanup_orphans()` at worker start | `test_orphaned_containers_are_cleaned_up` |
| GPUs | none visible unless `limits.gpu` (device request) | — (no GPU in S1.2) |
| PID 1 | `tini` (`init=True`) reaps stray processes | — |

Mount-sensitive tests run on two filesystems: pytest's `tmp_path` (WSL ext4 or Linux) and the
repo's git-ignored `runs/.sandbox-tests/`, which on Ghadi's machine is the Windows drive.

**Windows-drive differences (D-009).** On `/mnt/c`, Docker Desktop ignores Unix permissions on
bind mounts, so nothing here relies on them. It also refuses to create FIFOs, which is stricter.
The symlink, read-only, network, and limit behaviors are identical on both filesystems.

**Threat model, briefly.** Docker isolation is meaningful but not a guarantee against hostile code
(spec §14.3). Residual risks:
- kernel exploits that escape namespaces;
- side channels;
- on Windows, the shared Docker Desktop VM.

Hardening options for later: rootless Docker, a custom seccomp/AppArmor profile, gVisor or
microVMs. S3.D4's threat model expands on this.

## The image (`Dockerfile.tabular`)

- **Base:** `python:3.12-slim`, pinned by digest (same Python minor as the platform).
- **Packages:** installed with `pip --require-hashes --only-binary=:all: --no-deps` from
  `requirements-tabular.lock`. The lock is universal (all platforms) and hash-pinned, with releases
  no newer than `SANDBOX_EXCLUDE_NEWER` in the Makefile. The libraries are pandas, NumPy,
  scikit-learn, XGBoost (the CPU-only `xgboost-cpu` build), LightGBM, and Optuna.
- **Build context:** only the Dockerfile and the lock (`.dockerignore`).
- **User:** default non-root UID 10001, a second line of defense.
- **Tag:** `hypora-sandbox-tabular:<hash of Dockerfile + lock>`, so a changed recipe is never served
  from a stale image.
- **Commands:**
  - `make sandbox-image` builds the image.
  - After editing `requirements-tabular.in`, `make sandbox-lock` re-locks it (S1.3 adds its own
    libraries there).

## The job queue (`hypora/storage/jobs.py`, spec §14.2 / §15.3)

- **State machine:** spec §15.3, edge for edge (`ALLOWED_TRANSITIONS`). Every change is a
  compare-and-set on the expected state (and the lease owner, where one is required). Its
  `job_transition` event is written **in the same transaction**, so replay rebuilds job states from
  events alone.
- **`enqueue`:** DRAFT → VALIDATING → AWAITING_APPROVAL → QUEUED, idempotent per
  `idempotency_key`. The approval itself is S3.D2's; `approval_ref` is recorded.
- **`lease`:** one atomic `UPDATE … WHERE id = (oldest QUEUED job that fits the worker's free
  CPU/GPU/memory) AND status = 'QUEUED' … RETURNING`. Two workers can never lease the same job;
  tested with 6 threads and 4 processes. Job IDs are time-ordered, so "lowest ID" means oldest.
  Expired leases are requeued first, in the same transaction.
- **`heartbeat`:** extends the lease. It returns `LOST` (stop, leave the job alone) or
  `CANCEL_REQUESTED` (stop, then `confirm_cancelled`).
- **`complete` / `fail`:** by the lease holder only. A retryable failure is requeued while attempts
  remain (`max_attempts`); otherwise the job goes to `FAILED_TERMINAL`. Backoff per error class,
  idempotent duplicate completions, and crash detection are S3.E2.

## The worker (`hypora/execution/worker/`)

```
python -m hypora.execution.worker --db hypora.sqlite --planner package.module:planner [--once]
```

- **Leasing:** it leases only jobs that fit its free pools (`cpu_slots`, `gpu_slots`,
  `memory_budget` from S0.4; a job needs `cpu_threads`, or 1, plus its memory and one GPU if it asks)
  and runs each in its own thread.
- **Lifecycle per job:** plan → validate mounts → `start` (LEASED → RUNNING) → sandbox, with a
  heartbeat thread → `complete` / `fail` / `confirm_cancelled`.
- **Failure codes:** trial-caused failures are terminal: `EXIT_<n>`, `TIMEOUT`, `OOM_KILLED`,
  `OUTPUT_LIMIT`, `UNSAFE_OUTPUT`, `PLANNING_FAILED`. Infrastructure failures are retryable:
  `SANDBOX_START_FAILED`, `WORKER_ERROR`.
- **Planner:** a `JobPlanner` decides what runs (image, command, host directories); the worker
  decides how. S1.3 supplies the harness planner. A planner must give each job a fresh `/out`.
- **Startup:** at start the worker removes containers left by a crashed previous instance of
  itself.
