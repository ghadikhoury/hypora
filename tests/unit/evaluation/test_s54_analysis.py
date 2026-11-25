"""Unit tests for S5.4 consolidated analysis artifacts and evidence index."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = REPO_ROOT / "results" / "analysis"
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_s54_analysis.py"

CORE_ROWS = ("Agent", "Evaluation", "Analysis", "Repository")
CLAIMED_FEATURES = ("F1", "F3", "F4", "F5", "F6", "F7")
RUBRIC_PARTS = ("implementation", "architecture", "evaluation", "analysis")

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ghp_[0-9A-Za-z]{20,}"),
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_evidence_index_covers_all_core_and_feature_cells() -> None:
    index = _load_json(ANALYSIS_DIR / "evidence-index.json")
    core = index["core"]
    assert isinstance(core, dict)
    assert tuple(core.keys()) == CORE_ROWS
    for row_name, row in core.items():
        assert row["evidence_paths"], f"{row_name} missing evidence_paths"
        assert row["scripts"], f"{row_name} missing scripts"
        assert row["notes"], f"{row_name} missing notes"
        assert row["summary"], f"{row_name} missing summary"

    features = index["claimed_features"]
    assert isinstance(features, dict)
    assert tuple(features.keys()) == CLAIMED_FEATURES
    cell_count = 0
    for feat_id, feat in features.items():
        parts = feat["rubric_parts"]
        assert tuple(parts.keys()) == RUBRIC_PARTS
        for part_name, cell in parts.items():
            cell_count += 1
            assert cell["evidence_paths"], f"{feat_id}.{part_name} missing evidence_paths"
            assert cell["scripts"], f"{feat_id}.{part_name} missing scripts"
            assert cell["notes"], f"{feat_id}.{part_name} missing notes"
            assert cell["summary"], f"{feat_id}.{part_name} missing summary"
    assert cell_count == 24


def test_evidence_index_paths_all_exist_on_disk() -> None:
    index = _load_json(ANALYSIS_DIR / "evidence-index.json")
    referenced: set[str] = set()
    for row in index["core"].values():
        for key in ("evidence_paths", "scripts", "notes"):
            referenced.update(row[key])
    for feat in index["claimed_features"].values():
        for cell in feat["rubric_parts"].values():
            for key in ("evidence_paths", "scripts", "notes"):
                referenced.update(cell[key])

    missing = [rel for rel in sorted(referenced) if not (REPO_ROOT / rel).exists()]
    assert not missing, f"Missing paths referenced in evidence-index.json: {missing}"


def test_consolidated_summary_failure_catalog_and_determinism() -> None:
    summary = _load_json(ANALYSIS_DIR / "consolidated-summary.json")
    failures = summary["representative_failure_catalog"]
    assert isinstance(failures, list)
    assert 1 <= len(failures) <= 10
    for item in failures:
        assert item["case_ids"], f"Failure {item['category_id']} missing case_ids"
        for rel in item["evidence_paths"]:
            assert (REPO_ROOT / rel).exists(), f"Missing failure evidence path: {rel}"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"--check failed: {proc.stdout}\n{proc.stderr}"


def test_analysis_outputs_contain_no_secrets() -> None:
    for path in sorted(ANALYSIS_DIR.iterdir()):
        content = path.read_text(encoding="utf-8")
        for pat in SECRET_PATTERNS:
            assert not pat.search(content), f"Potential secret pattern {pat.pattern} in {path.name}"
