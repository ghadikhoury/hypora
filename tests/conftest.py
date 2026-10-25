"""Session fixtures shared across test packages."""

import pytest


@pytest.fixture(scope="session")
def sandbox_image() -> str:
    """The real tabular sandbox image (built on first use, about 2 minutes; then cached by tag)."""
    import docker

    from hypora.execution.sandbox.image import ensure_tabular_image

    return ensure_tabular_image(docker.from_env())
