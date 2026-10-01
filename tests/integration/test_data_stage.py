"""Integration tests for the data-stage files and schema behavior."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "styles_small.csv"


def run_data_stage(
    input_path: Path, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    """Run the data module with isolated configured output directories."""
    environment = os.environ.copy()
    environment.update(
        {
            "FASHION_INPUT": str(input_path),
            "FASHION_PROCESSED_DIR": str(tmp_path / "processed"),
            "FASHION_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        }
    )
    return subprocess.run(
        [sys.executable, "-m", "fashion_classifier.data_stage"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.integration
def test_data_stage_writes_auditable_outputs(tmp_path: Path) -> None:
    """Write clean, rejected, and profile outputs from only the fixture."""
    result = run_data_stage(FIXTURE, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Accepted rows: 35; rejected rows: 1" in result.stdout
    assert "Missing baseColour: 1" in result.stdout
    assert (tmp_path / "processed" / "clean.csv").is_file()
    assert (tmp_path / "processed" / "rejected_rows.csv").is_file()
    profile = json.loads(
        (tmp_path / "processed" / "profile.json").read_text(encoding="utf-8")
    )
    assert profile["accepted_rows"] == 35
    assert profile["rejected_rows"] == 1


@pytest.mark.integration
def test_data_stage_requires_base_colour(tmp_path: Path) -> None:
    """Fail clearly if the required target column is absent."""
    missing_target = tmp_path / "missing_target.csv"
    pd.DataFrame({"gender": ["Women"]}).to_csv(missing_target, index=False)

    result = run_data_stage(missing_target, tmp_path / "output")

    assert result.returncode == 1
    assert "Required column 'baseColour' is missing" in result.stderr


@pytest.mark.integration
def test_data_stage_adds_absent_optional_columns(tmp_path: Path) -> None:
    """Tolerate absent optional predictors and report the change."""
    minimal = tmp_path / "minimal.csv"
    pd.DataFrame({"baseColour": ["Black", "Blue"]}).to_csv(minimal, index=False)

    result = run_data_stage(minimal, tmp_path / "output")
    clean = pd.read_csv(tmp_path / "output" / "processed" / "clean.csv")

    assert result.returncode == 0, result.stderr
    assert "Optional predictor columns missing" in result.stderr
    assert "articleType" in clean.columns
    assert clean["articleType"].isna().all()
