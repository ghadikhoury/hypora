# Hypora

An agent-powered ML engineering platform for research datasets (CIS 1990 final project).
Provide a dataset, objective and constraints; Hypora runs bounded experiments and returns measured
results with reproducible artifacts and limitations.

**Release status.** The supported path is the local CLI for tabular classification and
regression. It includes the A/B/C workflow, plan approval, sandboxed experiments, a verified
result when one can be produced, and an inspectable trace. The single-agent and fixed-pipeline
systems remain available for evaluation. The [readiness records](docs/notes/readiness.md)
describe earlier development runs. Integrated post-hardening verification and the frozen
held-out evaluation are tracked in [the plan](docs/PLAN.md).

**Architecture.** The CLI hands the user objective and dataset reference to the controller.
A leads planning, delegation, and review. B gathers read-only research. C proposes data
methodology and writes trial code, optionally with bounded workers. The controller checks
approvals and budgets. Platform code locks splits and computes scores through the experiment
harness. The report and event log support review through `hypora status` and `hypora trace`.

See [the spec](docs/PROJECT_SPEC.md) for behavior and [AGENTS.md](AGENTS.md) for repository
workflow. Component guides live under [docs/notes/](docs/notes/).

## Requirements

- **Linux, or Windows 11 with WSL2** (Ubuntu 24.04). On Windows, run every command inside WSL,
  with the checkout on the WSL filesystem (`~/hypora`), never under `/mnt/c` (see Troubleshooting).
- **Docker**, running: Docker Desktop with WSL integration on Windows, Docker Engine on Linux.
- **git**, **make**, and **[uv](https://docs.astral.sh/uv/) 0.12.17 or newer**:
  `curl -LsSf https://astral.sh/uv/0.12.17/install.sh | sh`. uv installs the pinned Python (3.12)
  itself.
- **A Gemini API key** for the development model profile (free tier;
  [Google AI Studio](https://aistudio.google.com/apikey)).

## Setup

```bash
git clone https://github.com/ghadikhoury/hypora.git ~/hypora
cd ~/hypora
make setup                          # pinned Python + locked dependencies (uv), pre-commit hooks
cp .env.example .env && chmod 600 .env
```

Open `.env` in an editor and set `GEMINI_API_KEY=` to your key. `.env` is git-ignored: never commit
it or paste the key anywhere else.

## Run the agent

```bash
make run
```

The first time, this downloads the demo dataset (OpenML 46918, credit-g, CC BY 4.0) into
`data/snapshots/` and builds the sandbox image (about 2 minutes). It then runs the demo task
("predict whether a loan applicant is a good or bad credit risk") on the `dev` profile
(`gemini-3.5-flash-lite`, free tier). Progress prints as the run goes. The run ends with its
status, the final trial, and the path of its report (`runs/<run_id>/report.md`). The CLI exits
0 when the final model was verified, 1 when the run finishes without a verified model, and 2
when preflight fails. A run can take several minutes.

The run pauses at the plan gate and asks you to approve its plan and budget (or use
`uv run hypora approve <run_id>` from another terminal). For unattended runs add
`ARGS='--approve auto_within_contract'` to `make run` (or pass `--approve auto_within_contract` to
`hypora run`). `--advice-only` asks for research and advice only; nothing runs on the data.

Your own data (a CSV or Parquet file):

```bash
make run DATA=path/to/data.csv GOAL="Predict reaction yield; minimize MAE" TARGET=yield
uv run hypora run --data path/to/data.csv --goal "..." --target yield --max-trials 4
```

Inspect and control runs (state lives in `runs/`, which is git-ignored):

```bash
uv run hypora status                         # recent runs
uv run hypora status <run_id>                # status, spend, trials, jobs, final model, report
uv run hypora trace <run_id>                 # every event: role, type, duration, tokens
uv run hypora trace <run_id> -f type=model_call,tool_call -f role=agent_c
uv run hypora cancel <run_id>                # clean stop: cancels jobs, writes the report
uv run hypora --help
```

To keep a local copy of a completed report and rendered trace, follow the
[artifact export procedure](docs/notes/cli.md#local-artifact-export).

## Real datasets

Runs, evaluations, and demos use real OpenML or UCI datasets, stored as snapshots with a manifest
(source, version, license, retrieval date, SHA-256):

```bash
make snapshot SOURCE=openml ID=46918
make snapshot SOURCE=uci ID=17
uv run python scripts/snapshot_dataset.py list      # or: verify
```

Only datasets with a clearly named license are stored. Many older OpenML uploads declare just
`Public` and are refused, with a pointer to the UCI original when there is one. See
[docs/notes/datasets.md](docs/notes/datasets.md).

## Run the evaluation

For a release smoke check, fetch the pinned OpenML fish-toxicity dataset and grade the fixed
pipeline on the clean control. This command uses the sandbox and the same grader as a comparison,
without model calls.

```bash
make snapshot SOURCE=openml ID=46954
make eval CASES=dev SYSTEMS=baseline CONFIG=default PROFILE=dev \
  ARGS='--only dev_clean_control --results-root runs/release-smoke/results --work-root runs/release-smoke/work'
```

Read the path printed after `Results:` and open its `summary.md`. Check the per-case grade
there. An evaluator can complete successfully while a system fails a case. The smoke check
stores its outputs under git-ignored `runs/release-smoke/` so it can coexist with a later
comparison.

To compare the A/B/C agent and the fixed pipeline across all six development cases, first
[fetch the required snapshots and build the planted cases](hypora/evaluation/cases/dev/README.md).
Then run:

```bash
make eval CASES=dev SYSTEMS=agent,baseline CONFIG=default PROFILE=dev
```

The runner saves one primary result per case and system, a summary with confidence intervals
when applicable, and regression flags when compatible prior results exist. Run from a clean
checkout for commit-attributed records. An uncommitted development run can pass `--allow-dirty`
through `ARGS`, with results in `results/raw/`. This development check uses the free provider
profile. The [runner guide](docs/notes/eval-runner.md) covers paid profiles, bounded parallelism,
resume, and repeat samples.

## The Docker Compose stack (Mode 1)

```bash
make up            # API placeholder: http://127.0.0.1:8000/health (the real API is Later work)
make worker-up     # optional: the experiment worker as a compose service
make down
```

By default the agent and its experiment worker run on your machine as processes (`make run`).
`make worker-up` starts the worker as a container instead; use it with
`uv run hypora run --worker compose ...`. That container is the only one given the host's Docker
socket, which it needs to start sandboxes. [docs/notes/docker-compose.md](docs/notes/docker-compose.md)
explains why this is a platform-only privilege.

## Development

Run `make check-fast` (lint, types, and the in-process tests) before opening a PR. CI runs every
non-live test on code PRs and pushes to main, as parallel jobs behind one required `make check`
status; secret scanning is a second required check. Docs-only changes skip the tests.
[AGENTS.md](AGENTS.md) is the one place that describes the workflow, checks and review;
`make help` lists every target.

## Troubleshooting

- **`Docker is not reachable`**: start Docker Desktop (from WSL: `docker desktop start`) and
  enable integration for your distro in Docker Desktop → Settings → Resources → WSL integration.
  On Linux: `sudo systemctl start docker`.
- **`permission denied` on `/var/run/docker.sock`**: add yourself to the `docker` group
  (`sudo usermod -aG docker "$USER"`), then restart WSL (`wsl --shutdown` in PowerShell) or log in
  again.
- **Strange `getcwd` / "No such file or directory" failures, or very slow runs**: the checkout is
  under `/mnt/c`. Docker Desktop's bind mounts break processes working there, and the Windows drive
  is about 50x slower; clone into the WSL filesystem (`~/hypora`).
  [docs/notes/dev-environment.md](docs/notes/dev-environment.md) has the details.
- **`GEMINI_API_KEY is not set`**: `.env` must be in the repository root, with
  `GEMINI_API_KEY=<key>` on its own line.
- **The first run sits at "Building the sandbox image"**: expected once (about 2 minutes); the
  image is cached by a hash of its Dockerfile and lock file.
- **`make up` fails on port 8000**: another program uses it; stop it, or change the port in
  `docker-compose.yml`.
- **A run ends with "no verified model"** (exit code 1): read `runs/<run_id>/report.md` and
  `uv run hypora trace <run_id>`; the report says which stage stopped and why.
- **Files edited on Windows show `^M` or fail checks**: the repository uses LF line endings; edit
  through the WSL path or an editor set to LF.

## Testing the MCP server with the MCP Inspector

The platform's tools are served by a Model Context Protocol server (`hypora/mcp_server/`, spec
§12). To try them by hand in the official MCP Inspector, from WSL (needs Node.js 22 in WSL):

```bash
uv run --locked python -m hypora.mcp_server.dev            # dev workspace in runs/mcp-dev/
uv run --locked mcp dev hypora/mcp_server/dev_server.py     # terminal 1: opens the Inspector
uv run --locked python -m hypora.execution.worker.experiments \
  --db runs/mcp-dev/hypora.sqlite --runs-root runs/mcp-dev/runs   # terminal 2: runs experiments
```

Open the printed `http://127.0.0.1:6274?...` URL, click **Connect**, then **Tools → List Tools**.
The dev workspace uses the UCI Breast Cancer Wisconsin dataset bundled with scikit-learn.
A walkthrough of every tool, the call boundary, and the data firewall is in
[docs/notes/mcp-server.md](docs/notes/mcp-server.md).
