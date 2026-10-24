# Run from a POSIX shell: WSL2 Ubuntu on Windows, or Linux (CI). See AGENTS.md.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --locked

# Same gitleaks version as the pre-commit hook and the CI action; pinned by digest.
GITLEAKS_IMAGE := ghcr.io/gitleaks/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f

.PHONY: help setup lint typecheck test secrets check

help: ## List targets
	@grep -E '^[a-z]+:.*## ' Makefile | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install pinned Python + locked dev deps, install pre-commit hooks
	$(UV) sync --locked
	$(RUN) pre-commit install --install-hooks

lint: ## Ruff lint and format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck: ## mypy (strict on hypora/core/)
	$(RUN) mypy

test: ## pytest
	$(RUN) pytest

secrets: ## gitleaks over the full git history (requires Docker)
	docker run --rm -v "$(CURDIR):/repo:ro" $(GITLEAKS_IMAGE) git /repo --redact --no-banner --verbose

check: lint typecheck test secrets ## Everything that must pass before a PR
