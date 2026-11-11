# Hypora

An agent-powered ML engineering platform for research datasets (CIS 1990 final project).
Provide a dataset, objective and constraints; Hypora runs bounded experiments and returns measured
results with reproducible artifacts and limitations.

**Implemented:** the tabular single-agent pipeline and Stage 2 evaluation harness through PR #32.
The recorded six-case dev comparison passed 3/6 agent cases versus 2/6 fixed-baseline cases on each
profile; the comparison is inconclusive. Grouping, leakage and temporal methodology remain known gaps.

**Next product architecture (D-036):** A leads planning, delegation, review and reporting; B performs
read-only research, usually first; C owns data and coding and alone may request workers. Code enforces
approvals, budgets, isolation, splits and scoring. A can justify independent B/C overlap. This is the
implementation target; current single-agent runs remain the comparison system.

Read [the spec](docs/PROJECT_SPEC.md) for behavior, [the plan](docs/PLAN.md) for the remaining steps
and their dependencies, and [AGENTS.md](AGENTS.md) for how the repository is worked on. Component guides
live under `docs/notes/`; historical notes and results describe the version that produced them.

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
status, the final trial, and the path of its report (`runs/<run_id>/report.md`). The exit code is
0 when the final model was verified, 1 otherwise. A run takes a few minutes.

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
uv run hypora trace <run_id> -f type=model_call,tool_call -f role=single_agent
uv run hypora cancel <run_id>                # clean stop: cancels jobs, writes the report
uv run hypora --help
```

Details: [docs/notes/cli.md](docs/notes/cli.md).

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

```bash
make eval CASES=dev SYSTEMS=agent,baseline CONFIG=single_agent PROFILE=dev
```

The runner writes one primary result per case/system, a Markdown report with 95% intervals,
and newly failing cases compared with the nearest parent results for the same config. Named
configs live in `configs/eval/`; `PROFILE` overrides their model profile. `CASES` also accepts a
case directory. The tree must be clean for recorded results. See
[the runner guide](docs/notes/eval-runner.md) for bounded parallelism, resume, and repeat samples.
The first dev cases are added in S2.5; an empty case set is refused before calls.

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
