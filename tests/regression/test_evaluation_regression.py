"""Regression tests for evaluation report schemas and rare classes."""

from pathlib import Path

import pandas as pd
import pytest

from fashion_classifier.evaluation import evaluate_predictions


@pytest.mark.regression
def test_report_schema_low_support_and_pooled_oof(tmp_path: Path) -> None:
    """Keep class order, paired folds, and rare-class OOF rows visible."""
    cv_rows = []
    for fold in range(1, 6):
        fold_labels = ("Black", "Blue") if fold <= 3 else ("Black",)
        for offset, label in enumerate(fold_labels):
            row_id = fold * 10 + offset
            cv_rows.append(
                {
                    "row_id": row_id,
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
            "a_equivalent_prediction": ["Black", "Blue"],
            "b_prediction": ["Black", "Blue"],
        }
    ).to_csv(predictions_path, index=False)

    metrics = evaluate_predictions(predictions_path, cv_path, tmp_path / "artifacts")
    cv_report = pd.read_csv(tmp_path / "artifacts" / "cv_classification_report.csv")
    fold_report = pd.read_csv(tmp_path / "artifacts" / "cv_fold_metrics.csv")
    test_report = pd.read_csv(tmp_path / "artifacts" / "classification_report.csv")

    assert metrics["cv"]["mean_b_minus_a_macro_f1"] < 0
    assert fold_report["fold"].tolist() == [1, 2, 3, 4, 5]
    assert cv_report["class_label"].drop_duplicates().tolist() == ["Black", "Blue"]
    rare_class = cv_report.loc[cv_report["class_label"] == "Blue"]
    common_class = cv_report.loc[cv_report["class_label"] == "Black"]
    assert rare_class["low_support"].all()
    assert rare_class["support"].eq(3).all()
    assert not common_class["low_support"].any()
    assert cv_report.groupby("model")["support"].sum().to_dict() == {
        "a_equivalent": 8,
        "b": 8,
    }
    assert set(test_report["model"]) == {"majority", "a_equivalent", "b"}
    assert metrics["cv"]["pooled_oof_a_equivalent_macro_f1"] == pytest.approx(1.0)
