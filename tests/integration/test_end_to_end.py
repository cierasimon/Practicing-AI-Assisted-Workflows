"""End-to-end subprocess test for all user-facing pipeline stages."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "styles_small.csv"


@pytest.mark.integration
def test_all_stages_run_on_fixture_in_fresh_directories(tmp_path: Path) -> None:
    """Run every pipeline module using isolated fixture-backed artifacts."""
    processed_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"
    environment = os.environ.copy()
    environment.update(
        {
            "FASHION_INPUT": str(FIXTURE),
            "FASHION_PROCESSED_DIR": str(processed_dir),
            "FASHION_ARTIFACTS_DIR": str(artifact_dir),
            "FASHION_SEED": "42",
            "FASHION_TEST_SIZE": "0.2",
        }
    )
    for module in (
        "data_stage",
        "features_stage",
        "train_stage",
        "evaluate_stage",
    ):
        result = subprocess.run(
            [sys.executable, "-m", f"fashion_classifier.{module}"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{module}: {result.stdout}\n{result.stderr}"

    profile = json.loads((processed_dir / "profile.json").read_text(encoding="utf-8"))
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    cv = pd.read_csv(artifact_dir / "cv_predictions.csv")
    predictions = pd.read_csv(artifact_dir / "predictions.csv")
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    expected_labels = sorted(
        set(predictions["true_label"])
        | set(predictions["majority_prediction"])
        | set(predictions["a_equivalent_prediction"])
        | set(predictions["b_prediction"])
        | set(cv["true_label"])
        | set(cv["a_equivalent_prediction"])
        | set(cv["b_prediction"])
    )
    cv_delta = metrics["cv"]["mean_b_minus_a_macro_f1"]
    test_delta = metrics["test_metrics"]["b"]["macro_f1"] - metrics["test_metrics"]["a_equivalent"]["macro_f1"]
    if cv_delta > 0 and test_delta > 0:
        expected_decision = "improvement supported by positive CV and fixed-test macro-F1 deltas"
    elif cv_delta > 0 or test_delta > 0:
        expected_decision = "inconclusive: CV and fixed-test macro-F1 deltas disagree"
    else:
        expected_decision = "no improvement observed: both macro-F1 deltas are non-positive"

    assert profile["accepted_rows"] == 35
    assert profile["rejected_rows"] == 1
    assert profile["missing_baseColour"] == 1
    assert len(train) + len(test) == 34
    assert set(cv["row_id"]) == set(train["row_id"])
    assert set(cv["row_id"]).isdisjoint(test["row_id"])
    assert set(predictions["row_id"]) == set(test["row_id"])
    assert metrics["data_quality"]["rejected_rows"] == 1
    assert metrics["labels"] == expected_labels
    assert metrics["decision"] == expected_decision
    assert (artifact_dir / "comparison.md").is_file()


@pytest.mark.integration
def test_clean_stage_preserves_fixture_and_configured_input(tmp_path: Path) -> None:
    """Remove only generated output directories and keep raw fixture input."""
    processed_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"
    processed_dir.mkdir()
    artifact_dir.mkdir()
    (processed_dir / "generated.csv").write_text("generated\n", encoding="utf-8")
    (artifact_dir / "generated.json").write_text("{}", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "FASHION_INPUT": str(FIXTURE),
            "FASHION_PROCESSED_DIR": str(processed_dir),
            "FASHION_ARTIFACTS_DIR": str(artifact_dir),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "fashion_classifier.clean_stage"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not processed_dir.exists()
    assert not artifact_dir.exists()
    assert FIXTURE.is_file()
