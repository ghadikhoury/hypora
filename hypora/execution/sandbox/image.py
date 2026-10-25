"""Build the sandbox image, tagged by the hash of everything that goes into it.

The tag changes whenever the Dockerfile or the hash-locked requirements change, so a stale image
can never be used by mistake and an unchanged one is never rebuilt.
"""

from pathlib import Path
from typing import Any

import docker

from hypora.core.hashing import sha256_hex

SANDBOX_DIR = Path(__file__).resolve().parent
TABULAR_DOCKERFILE = "Dockerfile.tabular"
TABULAR_LOCK = "requirements-tabular.lock"
TABULAR_REPOSITORY = "hypora-sandbox-tabular"


def tabular_image_tag() -> str:
    """`hypora-sandbox-tabular:<first 16 hex of SHA-256(Dockerfile + lock)>`."""
    content = (SANDBOX_DIR / TABULAR_DOCKERFILE).read_bytes() + (
        SANDBOX_DIR / TABULAR_LOCK
    ).read_bytes()
    return f"{TABULAR_REPOSITORY}:{sha256_hex(content)[:16]}"


def ensure_tabular_image(client: Any | None = None) -> str:
    """Return the current tabular image tag, building it first if it does not exist locally."""
    client = client or docker.from_env()
    tag = tabular_image_tag()
    try:
        client.images.get(tag)
    except docker.errors.ImageNotFound:
        client.images.build(
            path=str(SANDBOX_DIR), dockerfile=TABULAR_DOCKERFILE, tag=tag, rm=True, pull=True
        )
    return tag


if __name__ == "__main__":
    print(ensure_tabular_image())
