"""Integration check for clear missing-input errors."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_data_stage_reports_missing_input(tmp_path: Path) -> None:
    """Run the data entry point with an isolated missing input path."""
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "FASHION_INPUT": str(tmp_path / "missing.csv"),
            "FASHION_PROCESSED_DIR": str(tmp_path / "processed"),
            "FASHION_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "fashion_classifier.data_stage"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Input file not found" in result.stderr
    assert (tmp_path / "processed").is_dir()
