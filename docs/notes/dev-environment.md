# Development environment: work from the WSL filesystem

**Work from the WSL filesystem (`~/hypora`), never from `/mnt/c`.**

Every working copy lives on WSL's own ext4 filesystem: `~/hypora` (Coding Agent A), a separate
clone per other agent, and temporary clones for reviews. Run `make`, `git`, and tests there.
Nothing Hypora runs should have its working directory on the Windows drive.

## Why (measured 2026-10-25, Docker Desktop 29.8 on WSL2)

- **Docker Desktop breaks working directories on `/mnt/c`.** When a container bind-mounts a path,
  Docker Desktop's WSL integration can make any other process whose working directory is on
  `/mnt/c` lose it: `getcwd()` fails with "No such file or directory", and child processes inherit
  the broken directory. It happened unpredictably, sometimes immediately and sometimes later, and
  broke 13 unrelated `make check` tests that start subprocesses. Processes on ext4 (or in `/`) were
  never affected. Docker Desktop also leaks these bind mounts (632 → 749 during the probes).
- **Permissions differ.** On `/mnt/c`, Docker Desktop ignores Unix permissions on bind mounts, and
  FIFOs cannot be created. On ext4 everything behaves like Linux CI (D-012).
- **Speed.** `/mnt/c` is a 9p network share inside WSL: a cold mypy run took about 10 minutes
  there.
- **Line endings.** Windows tools writing into the checkout can introduce CRLF; on ext4 the tools
  are Linux tools.

As defense in depth, the execution worker moves its own process to `/` at startup and uses absolute
paths only (`python -m hypora.execution.worker`), so it is unaffected wherever it is started.

## Setting up a working copy

```bash
# inside WSL (wsl -d Ubuntu-24.04)
git clone https://github.com/ghadikhoury/hypora.git ~/hypora      # or ~/hypora-<agent>
cd ~/hypora
git config user.name "Ghadi Khoury"
git config user.email "233380189+ghadikhoury@users.noreply.github.com"   # GitHub no-reply
git config core.autocrlf false
# Push with the Windows GitHub CLI login (no token is copied into WSL):
git config credential.https://github.com.helper ""
git config --add credential.https://github.com.helper \
  '!"/mnt/c/Program Files/GitHub CLI/gh.exe" auth git-credential'
make setup          # uv sync --locked + pre-commit hooks
make check
```

- **`.env` (API keys)** is per working copy and git-ignored. Copy it in yourself (never paste keys
  into a chat), e.g. `cp "/mnt/c/.../hypora/.env" ~/hypora/.env && chmod 600 ~/hypora/.env`.
- **GitHub CLI.** The `gh` commands (PRs, reviews) can run from Windows with
  `--repo ghadikhoury/hypora`, or from WSL after installing `gh` there and running `gh auth login`.
- **Windows editors** can open the WSL copy at `\\wsl.localhost\Ubuntu-24.04\home\ghadi\hypora`, or
  through VS Code's WSL extension (`code ~/hypora` inside WSL).
