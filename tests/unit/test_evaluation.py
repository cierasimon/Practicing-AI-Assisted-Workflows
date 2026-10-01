"""Unit tests for hand-checkable evaluation calculations."""

import pandas as pd
import pytest

from fashion_classifier.evaluation import calculate_metrics, compute_cv_fold_metrics


@pytest.mark.unit
def test_metrics_handle_a_class_with_no_predictions() -> None:
    """Return zero class metrics without undefined-division failures."""
    metrics = calculate_metrics(
        ["Black", "Black", "Blue"],
        ["Black", "Black", "Black"],
        ["Black", "Blue"],
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(0.4)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


@pytest.mark.unit
def test_cv_fold_scores_use_truth_classes_and_paired_deltas() -> None:
    """Compute fold metrics and paired score differences from OOF rows."""
    predictions = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "true_label": ["Black", "Blue", "Black", "Blue"],
            "a_equivalent_prediction": ["Black", "Blue", "Black", "Blue"],
            "b_prediction": ["Black", "Black", "Blue", "Blue"],
        }
    )

    scores = compute_cv_fold_metrics(predictions)

    assert scores["a_equivalent_macro_f1"].tolist() == [1.0, 1.0]
    assert scores["b_minus_a_macro_f1"].tolist() == pytest.approx([-2 / 3, -2 / 3])
