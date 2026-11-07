"""What the dev set's generation scripts share: a verified snapshot in, a planted file out (D-015).

A planted file is a pure function of its snapshot and its script: no clock, no global random state,
and the same Parquet writer as the snapshots, so the bytes reproduce for given library versions and
the hash in the case file can vouch for them. `data/` is git-ignored, so a machine regenerates them:

    uv run --locked python -m hypora.evaluation.cases.dev.planting
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from hypora.core.hashing import sha256_hex
from hypora.evaluation.case_data import PLANTED_DIR
from hypora.storage.snapshots import SNAPSHOTS_DIR, load_manifest, parquet_bytes, verify_entry


class PlantingError(RuntimeError):
    """The snapshot a script needs is missing or changed; nothing was written."""


@dataclass(frozen=True)
class Planting:
    """One generation script: the snapshot it reads, the file it writes, and the change it makes."""

    snapshot: str
    file: str
    plant: Callable[[pd.DataFrame], pd.DataFrame]

    def bytes(self, snapshots_dir: Path = SNAPSHOTS_DIR) -> bytes:
        """The planted file's content, from the snapshot the manifest vouches for."""
        entry = load_manifest(snapshots_dir).get(self.snapshot)
        if entry is None:
            raise PlantingError(f"snapshot {self.snapshot} is not in the manifest; see README.md")
        problem = verify_entry(entry, snapshots_dir)
        if problem is not None:
            raise PlantingError(problem)
        source = pd.read_parquet(snapshots_dir / entry.file, engine="pyarrow")
        content: bytes = parquet_bytes(self.plant(source))
        return content

    def write(
        self, snapshots_dir: Path = SNAPSHOTS_DIR, planted_dir: Path = PLANTED_DIR
    ) -> tuple[Path, str]:
        """Write the planted file; returns its path and SHA-256 (what the case file records)."""
        content = self.bytes(snapshots_dir)
        planted_dir.mkdir(parents=True, exist_ok=True)
        path = planted_dir / self.file
        path.write_bytes(content)
        return path, sha256_hex(content)


def main() -> None:
    """Regenerate every planted file of the dev set."""
    # Here rather than at module level: each script imports `Planting` from this module.
    from hypora.evaluation.cases.dev import plant_grouped_rows, plant_target_leakage

    for planting in (plant_target_leakage.PLANTING, plant_grouped_rows.PLANTING):
        path, digest = planting.write()
        print(f"{path}  {digest}")


if __name__ == "__main__":
    main()
