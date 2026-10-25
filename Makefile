# Run from a POSIX shell: WSL2 Ubuntu on Windows, or Linux (CI). See AGENTS.md.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --locked

# Same gitleaks version as the pre-commit hook and the CI action; pinned by digest.
GITLEAKS_IMAGE := ghcr.io/gitleaks/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f

.PHONY: help setup lint typecheck test test-live secrets schemas check

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' Makefile | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install pinned Python + locked dev deps, install pre-commit hooks
	$(UV) sync --locked
	$(RUN) pre-commit install --install-hooks

lint: ## Ruff lint and format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck: ## mypy (strict on hypora/core/)
	$(RUN) mypy

test: ## pytest (live provider calls excluded)
	$(RUN) pytest

test-live: ## Live LLM calls, free Gemini tier only (needs GEMINI_API_KEY in .env)
	$(RUN) pytest -m 'live and not paid' tests/live

secrets: ## gitleaks over the full git history (requires Docker)
	$(RUN) python scripts/scan_secrets.py --image "$(GITLEAKS_IMAGE)"

schemas: ## Regenerate docs/schemas/ from hypora.core.schemas
	$(RUN) python scripts/export_schemas.py

check: lint typecheck test secrets ## Everything that must pass before a PR
