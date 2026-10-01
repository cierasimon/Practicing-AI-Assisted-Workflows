"""Regression tests for deterministic training artifacts."""

from pathlib import Path

import pandas as pd
import pytest

from fashion_classifier.data import load_and_clean
from fashion_classifier.features import make_split
from fashion_classifier.modeling import fit_models


@pytest.mark.regression
def test_training_is_deterministic_and_oof_covers_training_only(tmp_path: Path) -> None:
    """Repeat fixture training and compare IDs, folds, and predictions."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "styles_small.csv"
    processed_dir = tmp_path / "processed"
    outputs = load_and_clean(fixture, processed_dir)
    train_path, test_path = make_split(outputs.clean_path, processed_dir, 0.2, 42)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_path = fit_models(train_path, test_path, first_dir, 42)
    second_path = fit_models(train_path, test_path, second_dir, 42)
    first_oof = pd.read_csv(first_dir / "cv_predictions.csv").sort_values("row_id")
    second_oof = pd.read_csv(second_dir / "cv_predictions.csv").sort_values("row_id")
    first_predictions = pd.read_csv(first_path).sort_values("row_id")
    second_predictions = pd.read_csv(second_path).sort_values("row_id")

    assert first_oof.equals(second_oof)
    assert first_predictions.equals(second_predictions)
    assert set(first_oof["row_id"]) == set(train["row_id"])
    assert first_oof["row_id"].is_unique
    assert set(first_oof["row_id"]).isdisjoint(test["row_id"])
    assert first_oof["fold"].nunique() == 5
