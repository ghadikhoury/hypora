# Run from a POSIX shell: WSL2 Ubuntu on Windows, or Linux (CI). See AGENTS.md.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --locked

# Same gitleaks version as the pre-commit hook and the CI action; pinned by digest.
GITLEAKS_IMAGE := ghcr.io/gitleaks/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f

# Sandbox package lock: newest releases at least this old (bump deliberately, then re-run sandbox-lock).
SANDBOX_EXCLUDE_NEWER := 2026-10-10
SANDBOX_DIR := hypora/execution/sandbox

# `make run` with no DATA runs the demo task on a real OpenML snapshot (D-015), fetched on first use:
# credit-g, OpenML 46918, CC BY 4.0 (D-021). DATA=<file> GOAL="<text>" [TARGET=<column>] runs your own.
DEMO_OPENML_ID := 46918
DEMO_DATA := data/snapshots/openml-46918-credit-g.parquet
DEMO_GOAL := Predict whether a loan applicant is a good or bad credit risk
DEMO_TARGET := good_or_bad_customer
DATA ?=
GOAL ?=
TARGET ?=
CASES ?= dev
SYSTEMS ?= agent,baseline
CONFIG ?= single_agent
PROFILE ?=
ARGS ?=

# Mode 1 compose stack (D-020). The worker file is the opt-in service holding the Docker socket.
COMPOSE := docker compose
WORKER_COMPOSE := $(COMPOSE) -f docker-compose.yml -f docker-compose.worker.yml

# Test selections (D-041). `test-fast` plus `test-containers` is exactly `test`: every non-live
# test either carries a container marker (`sandbox` or `docker`) or does not. CI runs each
# selection as parallel shards, split by path. In each family, `rest` ignores exactly the
# directories its sibling shards name, so the shards cover every test once and a new test
# directory lands in `rest` without editing this file (tests/unit/test_ci_shards.py checks this).
# Each expression repeats "not live" because a command-line -m replaces the one in pyproject.toml.
FAST_MARK := not live and not sandbox and not docker
CONTAINER_MARK := (sandbox or docker) and not live
SHARD ?= all
FAST_SHARD_all := tests
FAST_SHARD_agents := tests/controller tests/baseline
FAST_SHARD_rest := tests --ignore=tests/controller --ignore=tests/baseline
CONTAINER_SHARD_all := tests
CONTAINER_SHARD_controller := tests/controller
CONTAINER_SHARD_harness := tests/harness
CONTAINER_SHARD_rest := tests --ignore=tests/controller --ignore=tests/harness
PYTEST_ARGS ?=

.PHONY: help setup lint typecheck test test-fast test-containers test-live secrets schemas mcp-tools \
	sandbox-image sandbox-lock check check-fast check-ci run snapshot eval up worker-up down

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' Makefile | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

run: ## Run the agent: the demo task, or DATA=<file> GOAL="<text>" [TARGET=<col>] [ARGS=...]
ifeq ($(strip $(DATA)),)
	$(RUN) python scripts/snapshot_dataset.py openml $(DEMO_OPENML_ID)
	$(RUN) hypora run --data "$(DEMO_DATA)" --goal "$(DEMO_GOAL)" --target "$(DEMO_TARGET)" $(ARGS)
else
	@test -n "$(GOAL)" || { echo 'make run DATA=... also needs GOAL="<the objective>"' >&2; exit 2; }
	$(RUN) hypora run --data "$(DATA)" --goal "$(GOAL)" $(if $(TARGET),--target "$(TARGET)") $(ARGS)
endif

snapshot: ## Fetch a real dataset into data/snapshots/: make snapshot SOURCE=openml|uci ID=<id>
	@test -n "$(SOURCE)" && test -n "$(ID)" \
		|| { echo 'usage: make snapshot SOURCE=openml|uci ID=<dataset id>' >&2; exit 2; }
	$(RUN) python scripts/snapshot_dataset.py $(SOURCE) $(ID)

eval: ## Compare systems: CASES=dev SYSTEMS=agent,baseline CONFIG=single_agent [PROFILE=dev]
	$(RUN) hypora eval --cases "$(CASES)" --systems "$(SYSTEMS)" --config "$(CONFIG)" \
		$(if $(PROFILE),--profile "$(PROFILE)") $(ARGS)

up: ## Mode 1 compose stack: the API placeholder on http://127.0.0.1:8000/health (needs Docker)
	mkdir -p runs
	RUNS_DIR="$(CURDIR)/runs" $(COMPOSE) up -d --build --wait

worker-up: ## Opt-in compose experiment worker, for `hypora run --worker compose` (D-020)
	mkdir -p runs
	$(RUN) python -m hypora.storage.migrate runs/hypora.sqlite
	REPO_DIR="$(CURDIR)" RUNS_DIR="$(CURDIR)/runs" WORKER_UID="$$(id -u)" WORKER_GID="$$(id -g)" \
		DOCKER_GID="$$(stat -c %g /var/run/docker.sock)" $(WORKER_COMPOSE) up -d --build --wait worker

down: ## Stop the compose stack, the opt-in worker included
	$(COMPOSE) down --remove-orphans

setup: ## Install pinned Python + locked dev deps, install pre-commit hooks
	$(UV) sync --locked
	$(RUN) pre-commit install --install-hooks

lint: ## Ruff lint and format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck: ## mypy (strict on hypora/core/)
	$(RUN) mypy

test: ## pytest, every non-live test (live provider calls excluded)
	$(RUN) pytest $(PYTEST_ARGS)

test-fast: ## Tests that start no container: [SHARD=all|agents|rest]; agents = scripted agent loops
	@test -n "$(FAST_SHARD_$(SHARD))" \
		|| { echo 'usage: make test-fast [SHARD=all|agents|rest]' >&2; exit 2; }
	$(RUN) pytest -m "$(FAST_MARK)" $(FAST_SHARD_$(SHARD)) $(PYTEST_ARGS)

test-containers: ## Tests that start containers: [SHARD=all|controller|harness|rest] (needs Docker)
	@test -n "$(CONTAINER_SHARD_$(SHARD))" \
		|| { echo 'usage: make test-containers [SHARD=all|controller|harness|rest]' >&2; exit 2; }
	$(RUN) pytest -m "$(CONTAINER_MARK)" $(CONTAINER_SHARD_$(SHARD)) $(PYTEST_ARGS)

test-live: ## Live LLM calls, free Gemini tier only (needs GEMINI_API_KEY in .env)
	$(RUN) pytest -m 'live and not paid' tests/live

secrets: ## gitleaks over the full git history (requires Docker)
	$(RUN) python scripts/scan_secrets.py --image "$(GITLEAKS_IMAGE)"

schemas: ## Regenerate docs/schemas/ from hypora.core.schemas
	$(RUN) python scripts/export_schemas.py

mcp-tools: ## Regenerate docs/mcp/tools.json, the frozen MCP tool contract (review the diff)
	$(RUN) python scripts/export_mcp_tools.py

sandbox-image: ## Build the tabular sandbox image (tag = hash of Dockerfile + lock; needs Docker)
	$(RUN) python -m hypora.execution.sandbox.image

sandbox-lock: ## Re-lock the sandbox image's packages (hashes, all platforms) after editing the .in
	cd $(SANDBOX_DIR) && $(UV) pip compile requirements-tabular.in --universal --generate-hashes \
		--python-version 3.12 --exclude-newer $(SANDBOX_EXCLUDE_NEWER) --no-header --quiet \
		-o requirements-tabular.lock

# The in-process tests only: no containers, and not the scripted agent loops (SHARD=agents), which
# start an MCP server process per test and take as long again. `make check-fast SHARD=all` adds them.
check-fast: SHARD := rest
check-fast: lint typecheck test-fast ## Before opening a PR: lint, types, in-process tests

check: lint typecheck test secrets ## Full local validation plus a full-history secret scan (slow)

check-ci: lint typecheck test ## Everything CI covers, in one serial run (CI runs it as parallel jobs)
