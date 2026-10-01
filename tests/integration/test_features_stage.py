"""Integration tests for the feature-stage module boundary."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_features_stage_adds_missing_predictors_and_writes_splits(
    tmp_path: Path,
) -> None:
    """Run the module on clean data without optional feature columns."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame({"row_id": range(10), "baseColour": ["Black", "Blue"] * 5}).to_csv(
        processed_dir / "clean.csv", index=False
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FASHION_PROCESSED_DIR": str(processed_dir),
            "FASHION_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "fashion_classifier.features_stage"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    assert (processed_dir / "train.csv").is_file()
    assert (processed_dir / "test.csv").is_file()
    assert len(test) == 2
    assert set(train["row_id"]).isdisjoint(test["row_id"])
    assert train["year"].isna().all()
    assert "productDisplayName" not in train.columns
