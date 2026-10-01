"""Integration tests for evaluation artifacts and conclusion policy."""

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion_classifier.data import load_and_clean
from fashion_classifier.evaluation import add_data_quality_context, evaluate_predictions
from fashion_classifier.features import make_split
from fashion_classifier.modeling import fit_models


@pytest.mark.integration
def test_evaluate_fixture_artifacts_and_data_quality(tmp_path: Path) -> None:
    """Evaluate real fixture model outputs and write every report artifact."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "styles_small.csv"
    processed_dir = tmp_path / "processed"
    outputs = load_and_clean(fixture, processed_dir)
    train_path, test_path = make_split(outputs.clean_path, processed_dir, 0.2, 42)
    artifact_dir = tmp_path / "artifacts"
    predictions_path = fit_models(train_path, test_path, artifact_dir, 42)

    metrics = evaluate_predictions(
        predictions_path,
        artifact_dir / "cv_predictions.csv",
        artifact_dir,
    )
    add_data_quality_context(artifact_dir, outputs.profile_path)
    written = {path.name for path in artifact_dir.iterdir() if path.is_file()}
    confusion = pd.read_csv(artifact_dir / "confusion_matrix.csv")
    test_report = pd.read_csv(artifact_dir / "classification_report.csv")
    persisted_metrics = json.loads(
        (artifact_dir / "metrics.json").read_text(encoding="utf-8")
    )

    assert {
        "metrics.json",
        "cv_fold_metrics.csv",
        "cv_classification_report.csv",
        "classification_report.csv",
        "confusion_matrix.csv",
        "comparison.md",
    }.issubset(written)
    assert confusion.groupby("model")["count"].sum().to_dict() == {
        "a_equivalent": 7,
        "b": 7,
        "majority": 7,
    }
    assert test_report.groupby("model")["support"].sum().to_dict() == {
        "a_equivalent": 7,
        "b": 7,
        "majority": 7,
    }
    assert persisted_metrics["data_quality"]["missing_baseColour"] == 1
    assert (
        "historical repo a"
        in (artifact_dir / "comparison.md").read_text(encoding="utf-8").lower()
    )
    comparison = (artifact_dir / "comparison.md").read_text(encoding="utf-8").lower()
    a_metrics = metrics["test_metrics"]["a_equivalent"]
    b_metrics = metrics["test_metrics"]["b"]

    def metric_direction(value: float, reference: float) -> str:
        if value > reference:
            return "higher"
        if value < reference:
            return "lower"
        return "equal"

    assert (
        "on this fixed test split, b's accuracy was "
        f"{metric_direction(b_metrics['accuracy'], a_metrics['accuracy'])} "
        "than a-equivalent's, while b's macro f1 was "
        f"{metric_direction(b_metrics['macro_f1'], a_metrics['macro_f1'])} "
        "and its balanced accuracy was "
        f"{metric_direction(b_metrics['balanced_accuracy'], a_metrics['balanced_accuracy'])}."
    ) in comparison
    assert metrics["labels"]


@pytest.mark.integration
def test_disagreeing_cv_and_test_deltas_are_inconclusive(tmp_path: Path) -> None:
    """Do not claim improvement when paired CV and test evidence disagree."""
    cv_rows = []
    for fold in range(1, 6):
        for offset, label in enumerate(("Black", "Blue")):
            cv_rows.append(
                {
                    "row_id": fold * 10 + offset,
                    "fold": fold,
                    "true_label": label,
                    "a_equivalent_prediction": label,
                    "b_prediction": "Black",
                }
            )
    cv_path = tmp_path / "cv.csv"
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False)
    pd.DataFrame(
        {
            "row_id": [100, 101],
            "true_label": ["Black", "Blue"],
            "majority_prediction": ["Black", "Black"],
            "a_equivalent_prediction": ["Blue", "Black"],
            "b_prediction": ["Black", "Blue"],
        }
    ).to_csv(predictions_path, index=False)

    metrics = evaluate_predictions(predictions_path, cv_path, tmp_path / "artifacts")

    assert metrics["cv"]["mean_b_minus_a_macro_f1"] < 0
    assert (
        metrics["test_metrics"]["b"]["macro_f1"]
        > metrics["test_metrics"]["a_equivalent"]["macro_f1"]
    )
    assert metrics["decision"].startswith("inconclusive")
