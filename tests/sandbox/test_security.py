"""Sandbox security tests (PLAN.md S1.2, spec §14.3, D-012) against the real image.

Each test runs untrusted-style code in a container and checks that the boundary holds. Tests that
depend on mounts run on both filesystems (`base_dir`: tmp and the repo's runs/, i.e. /mnt/c on
Ghadi's machine). They must pass locally on WSL2 and in CI (Linux).
"""

import contextlib
import json
import os
import time
from pathlib import Path

import docker
import pytest

from hypora.execution.sandbox.launcher import LABEL_JOB, SandboxLauncher, SandboxStatus
from hypora.execution.sandbox.outputs import UnsafeOutputError, open_output
from hypora.execution.sandbox.spec import (
    SandboxConfigError,
    SandboxLimits,
    SandboxMounts,
    SandboxSpec,
)
from tests.sandbox.conftest import RunScript, make_mounts

pytestmark = pytest.mark.sandbox

PROBE = r"""
import json, os, sys
def attempt(fn):
    try:
        fn()
        return "allowed"
    except Exception as e:
        return f"blocked:{type(e).__name__}"
"""


def read_json(path: Path) -> dict[str, object]:
    with open_output(path.parent, path.name) as handle:
        loaded: dict[str, object] = json.loads(handle.read())
    return loaded


# --- identity and privileges ----------------------------------------------------------------------


def test_runs_as_the_workers_non_root_uid_without_capabilities(
    run_script: RunScript, tmp_path: Path
) -> None:
    code = (
        PROBE
        + r"""
status = dict(line.split(":", 1) for line in open("/proc/self/status").read().splitlines())
json.dump({
    "uid": os.getuid(), "gid": os.getgid(),
    "cap_eff": status["CapEff"].strip(), "cap_prm": status["CapPrm"].strip(),
    "cap_bnd": status["CapBnd"].strip(), "no_new_privs": status["NoNewPrivs"].strip(),
    "seccomp": status["Seccomp"].strip(),
    "become_root": attempt(lambda: os.setuid(0)),
}, open("/out/identity.json", "w"))
"""
    )
    mounts, result = run_script(code, tmp_path)
    assert result.status is SandboxStatus.COMPLETED, result.detail + result.logs
    identity = read_json(mounts.out / "identity.json")
    assert identity["uid"] == os.getuid() != 0 and identity["gid"] == os.getgid()
    assert identity["cap_eff"] == identity["cap_prm"] == identity["cap_bnd"] == "0000000000000000"
    assert identity["no_new_privs"] == "1"
    assert identity["seccomp"] == "2"  # Docker's default seccomp filter is active
    assert str(identity["become_root"]).startswith("blocked")


# --- network --------------------------------------------------------------------------------------


def test_network_access_fails(run_script: RunScript, tmp_path: Path) -> None:
    code = (
        PROBE
        + r"""
import socket, urllib.request
json.dump({
    "tcp": attempt(lambda: socket.create_connection(("1.1.1.1", 53), timeout=3)),
    "dns": attempt(lambda: socket.getaddrinfo("example.com", 443)),
    "http": attempt(lambda: urllib.request.urlopen("http://example.com", timeout=3)),
    "interfaces": sorted(os.listdir("/sys/class/net")),
}, open("/out/net.json", "w"))
"""
    )
    mounts, result = run_script(code, tmp_path)
    assert result.status is SandboxStatus.COMPLETED, result.detail + result.logs
    net = read_json(mounts.out / "net.json")
    assert all(str(net[k]).startswith("blocked") for k in ("tcp", "dns", "http")), net
    assert net["interfaces"] == ["lo"]


# --- filesystem -----------------------------------------------------------------------------------


def test_writes_outside_trial_and_out_fail(run_script: RunScript, base_dir: Path) -> None:
    code = (
        PROBE
        + r"""
targets = ["/data/evil.csv", "/data/train.csv", "/harness/harness.py", "/evil", "/usr/evil",
           "/etc/passwd", "/root/evil", "/home/evil"]
results = {t: attempt(lambda t=t: open(t, "a").write("x")) for t in targets}
results["/trial/model.bin"] = attempt(lambda: open("/trial/model.bin", "w").write("x"))
results["/out/result.json"] = attempt(lambda: open("/out/result.json", "w").write("{}"))
results["/tmp/scratch"] = attempt(lambda: open("/tmp/scratch", "w").write("x"))
json.dump(results, open("/out/writes.json", "w"))
"""
    )
    mounts, result = run_script(code, base_dir)
    assert result.status is SandboxStatus.COMPLETED, result.detail + result.logs
    writes = read_json(mounts.out / "writes.json")
    for target in (
        "/data/evil.csv",
        "/data/train.csv",
        "/harness/harness.py",
        "/evil",
        "/usr/evil",
        "/etc/passwd",
        "/root/evil",
        "/home/evil",
    ):
        assert str(writes[target]).startswith("blocked"), (target, writes[target])
    for target in ("/trial/model.bin", "/out/result.json", "/tmp/scratch"):
        assert writes[target] == "allowed", (target, writes[target])
    assert (mounts.data / "train.csv").read_text(encoding="utf-8") == "x,y\n1,2\n"
    assert sorted(p.name for p in mounts.data.iterdir()) == ["train.csv"]
    assert (mounts.harness / "harness.py").read_text(encoding="utf-8") == "# platform code\n"


def test_non_mounted_host_paths_are_invisible(run_script: RunScript, base_dir: Path) -> None:
    secret = base_dir / "host-secret.txt"  # sibling of the mounts, never mounted
    secret.write_text("HOST SECRET", encoding="utf-8")
    home = Path.home()
    code = (
        PROBE
        + f"""
paths = [{str(secret)!r}, {str(home)!r}, "/var/run/docker.sock", "/run/docker.sock", "/mnt/c",
         "/proc/1/root/etc/shadow"]
json.dump({{p: attempt(lambda p=p: os.listdir(p) if os.path.isdir(p) else open(p).read())
           for p in paths}}, open("/out/reads.json", "w"))
"""
    )
    mounts, result = run_script(code, base_dir)
    assert result.status is SandboxStatus.COMPLETED, result.detail + result.logs
    reads = read_json(mounts.out / "reads.json")
    assert all(str(v).startswith("blocked") for v in reads.values()), reads
    assert "HOST SECRET" not in result.logs


def test_tmp_is_size_capped_and_not_executable(run_script: RunScript, tmp_path: Path) -> None:
    code = (
        PROBE
        + r"""
import subprocess
script = "/tmp/run.sh"
open(script, "w").write("#!/bin/sh\necho pwned\n")
os.chmod(script, 0o755)
json.dump({
    "exec_from_tmp": attempt(lambda: subprocess.run([script], check=True)),
    "fill_tmp": attempt(lambda: open("/tmp/big", "wb").write(b"0" * (8 * 1024 * 1024))),
}, open("/out/tmp.json", "w"))
"""
    )
    mounts, result = run_script(code, tmp_path, tmpfs_bytes=4 * 1024 * 1024)
    assert result.status is SandboxStatus.COMPLETED, result.detail + result.logs
    tmp = read_json(mounts.out / "tmp.json")
    assert str(tmp["exec_from_tmp"]).startswith("blocked:PermissionError")
    assert str(tmp["fill_tmp"]).startswith("blocked")


def test_worker_environment_never_leaks(
    run_script: RunScript, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "should-never-appear")
    code = 'import json, os; json.dump(dict(os.environ), open("/out/env.json", "w"))'
    mounts, result = run_script(code, tmp_path, env={"JOB_SETTING": "visible"})
    assert result.status is SandboxStatus.COMPLETED, result.detail + result.logs
    env = read_json(mounts.out / "env.json")
    assert "FAKE_PROVIDER_API_KEY" not in env and "should-never-appear" not in json.dumps(env)
    assert env["JOB_SETTING"] == "visible" and env["HOME"] == "/tmp"


# --- the symlink escape (D-012; also an attack case for S3.D4) ------------------------------------


def test_symlink_escape_through_outputs_is_refused(run_script: RunScript, base_dir: Path) -> None:
    secret = base_dir / "host-secret.txt"
    secret.write_text("HOST SECRET", encoding="utf-8")
    code = f"""
import os
os.symlink("/etc/passwd", "/out/model.bin")                 # absolute, container view
os.symlink({str(secret)!r}, "/out/metrics.json")             # absolute, host view
os.symlink("../host-secret.txt", "/trial/predictions.csv")   # relative escape from /trial
os.mkdir("/out/dir")
os.symlink("/", "/out/dir/root")                             # link to a directory
open("/out/real.json", "w").write("{{}}")
try:
    os.mkfifo("/out/pipe")                                   # special file
except OSError as e:  # the Windows drive (Docker Desktop share) refuses FIFOs outright
    open("/out/fifo-refused.txt", "w").write(str(e.errno))
"""
    mounts, result = run_script(code, base_dir)
    assert result.status is SandboxStatus.UNSAFE_OUTPUT, result.detail + result.logs
    assert result.exit_code == 0
    assert result.outputs is not None
    fifo_created = "fifo-refused.txt" not in {f.path for f in result.outputs.files}
    # Only the Windows drive may refuse FIFOs; everywhere else the rejection path must be exercised.
    assert fifo_created or str(base_dir).startswith("/mnt/")
    expected = ["/out/model.bin", "/out/metrics.json", "/trial/predictions.csv", "/out/dir/root"]
    if fifo_created:
        expected.append("/out/pipe")
    for name in expected:
        assert name in result.detail
    # Every link and special file was unlinked on the host; the targets are untouched.
    leftovers = [p for p in (*mounts.out.rglob("*"), *mounts.trial.rglob("*")) if p.is_symlink()]
    assert leftovers == [] and not (mounts.out / "pipe").exists()
    assert secret.read_text(encoding="utf-8") == "HOST SECRET"
    assert "HOST SECRET" not in result.detail + result.logs
    # Only regular files remain in the manifest.
    assert {f.path for f in result.outputs.files} <= {"real.json", "fifo-refused.txt"}


def test_open_output_refuses_links_left_in_a_mount(base_dir: Path) -> None:
    mounts = make_mounts(base_dir)
    (base_dir / "host-secret.txt").write_text("HOST SECRET", encoding="utf-8")
    (mounts.out / "model.bin").symlink_to(base_dir / "host-secret.txt")
    with pytest.raises(UnsafeOutputError):
        open_output(mounts.out, "model.bin")


# --- limits ---------------------------------------------------------------------------------------


def test_timeout_kills_the_job(
    run_script: RunScript, tmp_path: Path, launcher: SandboxLauncher
) -> None:
    started = time.monotonic()
    _, result = run_script(
        "import time; time.sleep(120)", tmp_path, job_id="job_timeout", wall_seconds=3
    )
    assert result.status is SandboxStatus.TIMEOUT
    assert time.monotonic() - started < 45
    remaining = launcher.client.containers.list(
        all=True, filters={"label": f"{LABEL_JOB}=job_timeout"}
    )
    assert remaining == []  # killed AND removed


def test_memory_limit_kills_the_job(run_script: RunScript, tmp_path: Path) -> None:
    code = "chunks = []\nwhile True:\n    chunks.append(bytearray(32 * 1024 * 1024))\n"
    _, result = run_script(code, tmp_path, memory_bytes=128 * 1024**2)
    assert result.status is SandboxStatus.OOM_KILLED, result.detail


def test_process_limit_contains_a_fork_bomb(run_script: RunScript, tmp_path: Path) -> None:
    code = r"""
import json, subprocess
procs = []
try:
    for _ in range(500):
        procs.append(subprocess.Popen(["sleep", "30"]))
except OSError:
    pass
json.dump({"spawned": len(procs)}, open("/out/pids.json", "w"))
for p in procs:
    p.kill()
"""
    mounts, result = run_script(code, tmp_path, pids=32)
    assert result.status is SandboxStatus.COMPLETED, result.detail + result.logs
    spawned = read_json(mounts.out / "pids.json")["spawned"]
    assert isinstance(spawned, int) and spawned < 32


def test_single_file_size_is_capped(run_script: RunScript, tmp_path: Path) -> None:
    code = r"""
import json
try:
    with open("/out/huge.bin", "wb") as f:
        f.write(b"0" * (3 * 1024 * 1024))
    outcome = "written"
except OSError as e:
    outcome = f"blocked:{e.errno}"
import os
os.remove("/out/huge.bin")
json.dump({"outcome": outcome}, open("/out/fsize.json", "w"))
"""
    mounts, result = run_script(code, tmp_path, max_file_bytes=1024 * 1024)
    assert result.status is SandboxStatus.COMPLETED, result.detail + result.logs
    assert read_json(mounts.out / "fsize.json")["outcome"] == "blocked:27"  # EFBIG


def test_total_output_is_capped(run_script: RunScript, tmp_path: Path) -> None:
    code = r"""
import time
for i in range(200):
    open(f"/out/part{i}.bin", "wb").write(b"0" * (1024 * 1024))
    time.sleep(0.05)
"""
    _, result = run_script(code, tmp_path, output_bytes=8 * 1024 * 1024, max_file_bytes=2 * 1024**2)
    assert result.status is SandboxStatus.OUTPUT_LIMIT, result.detail


def test_abort_request_kills_the_job(run_script: RunScript, tmp_path: Path) -> None:
    deadline = time.monotonic() + 2
    _, result = run_script(
        "import time; time.sleep(120)",
        tmp_path,
        abort=lambda: "lease lost" if time.monotonic() > deadline else None,
    )
    assert result.status is SandboxStatus.ABORTED and result.detail == "lease lost"


def test_failing_job_reports_its_exit_code_and_logs(run_script: RunScript, tmp_path: Path) -> None:
    _, result = run_script("print('hello from trial'); raise SystemExit(3)", tmp_path)
    assert result.status is SandboxStatus.FAILED and result.exit_code == 3
    assert "hello from trial" in result.logs


# --- refusing unsafe specs before anything starts -------------------------------------------------


def spec_for(mounts: SandboxMounts) -> SandboxSpec:
    return SandboxSpec(
        job_id="job_bad",
        worker_id="w-test",
        image="hypora-sandbox-tabular:unused",
        command=("true",),
        mounts=mounts,
        limits=SandboxLimits(wall_seconds=5, memory_bytes=64 * 1024**2, cpus=1),
    )


def test_overlapping_mounts_are_refused(tmp_path: Path) -> None:
    mounts = make_mounts(tmp_path)
    inside = mounts.trial / "data"
    inside.mkdir()
    with pytest.raises(SandboxConfigError, match="overlap"):
        SandboxMounts(
            data=inside, trial=mounts.trial, harness=mounts.harness, out=mounts.out
        ).validated()
    harness_in_trial = mounts.trial / "harness"
    harness_in_trial.mkdir()
    with pytest.raises(SandboxConfigError, match="overlap"):
        SandboxMounts(
            data=mounts.data, trial=mounts.trial, harness=harness_in_trial, out=mounts.out
        ).validated()
    out_in_trial = mounts.trial / "out"
    out_in_trial.mkdir()
    with pytest.raises(SandboxConfigError, match="overlap"):
        SandboxMounts(
            data=mounts.data, trial=mounts.trial, harness=mounts.harness, out=out_in_trial
        ).validated()


def test_a_symlinked_mount_source_is_resolved_before_the_overlap_check(tmp_path: Path) -> None:
    mounts = make_mounts(tmp_path)
    alias = tmp_path / "innocent-looking"
    alias.symlink_to(mounts.trial / ".", target_is_directory=True)
    with pytest.raises(SandboxConfigError, match="overlap"):
        SandboxMounts(
            data=alias, trial=mounts.trial, harness=mounts.harness, out=mounts.out
        ).validated()


def test_out_must_be_fresh_and_paths_must_exist(tmp_path: Path, launcher: SandboxLauncher) -> None:
    mounts = make_mounts(tmp_path)
    (mounts.out / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SandboxConfigError, match="fresh, empty"):
        launcher.run(spec_for(mounts))
    missing = SandboxMounts(
        data=tmp_path / "nope", trial=mounts.trial, harness=mounts.harness, out=mounts.out
    )
    with pytest.raises(SandboxConfigError, match="not an existing directory"):
        missing.validated()


# --- housekeeping ---------------------------------------------------------------------------------


def test_orphaned_containers_are_cleaned_up(sandbox_image: str, launcher: SandboxLauncher) -> None:
    client = docker.from_env()
    orphan = client.containers.run(
        sandbox_image,
        ["sleep", "300"],
        detach=True,
        network_mode="none",
        labels={
            "hypora.sandbox": "1",
            "hypora.worker": "w-crashed",
            "hypora.job": "job_orphan",
            "hypora.deadline": str(int(time.time()) + 3600),
        },
    )
    try:
        removed = launcher.cleanup_orphans("w-crashed")
        assert orphan.id in removed
        assert client.containers.list(all=True, filters={"id": orphan.id}) == []
    finally:
        with contextlib.suppress(docker.errors.NotFound):
            orphan.remove(force=True)


def test_no_sandbox_container_outlives_its_run(
    run_script: RunScript, tmp_path: Path, launcher: SandboxLauncher
) -> None:
    run_script("print('done')", tmp_path, job_id="job_lifetime")
    assert (
        launcher.client.containers.list(all=True, filters={"label": f"{LABEL_JOB}=job_lifetime"})
        == []
    )
