# Hypora platform image (PLAN.md S1.6, spec §4.1): the services of docker-compose.yml, i.e. the API
# placeholder and the opt-in experiment worker (D-020). Platform code only: LLM-written trial code
# never runs here, it runs in the sandbox image (hypora/execution/sandbox/Dockerfile.tabular).
#
# Build context: the repository root, filtered by .dockerignore to exactly the files below, so
# .env, runs/, data/, and the git history never enter the image.

# python:3.12-slim (Debian), pinned by digest; the same base as the sandbox image.
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

# uv, pinned by digest to the version CI and pyproject require (0.12.17).
COPY --from=ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc /uv /usr/local/bin/uv

# The interpreter is the exact one .python-version pins (as in development and CI): uv installs
# that release (checksum-verified by uv) under /opt/python instead of using the base image's 3.12.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Locked runtime dependencies first (no dev group), so code changes do not reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY configs/ configs/
COPY hypora/ hypora/
RUN uv sync --locked --no-dev --no-cache

# Non-root by default. The compose worker runs as the host user instead (docker-compose.yml).
RUN groupadd --gid 10001 hypora \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin hypora
USER 10001:10001

CMD ["python", "-m", "hypora.api", "--host", "0.0.0.0", "--port", "8000", "--state-dir", "/state"]
