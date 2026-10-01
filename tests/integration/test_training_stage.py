"""Integration tests for saved models and prediction artifacts."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from fashion_classifier.data import load_and_clean
from fashion_classifier.features import make_split
from fashion_classifier.modeling import fit_models


@pytest.mark.integration
def test_fit_save_load_predicts_unseen_category_and_nulls(tmp_path: Path) -> None:
    """Save/load model pipelines and predict every persisted test row."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "styles_small.csv"
    processed_dir = tmp_path / "processed"
    outputs = load_and_clean(fixture, processed_dir)
    train_path, test_path = make_split(outputs.clean_path, processed_dir, 0.2, 42)
    test = pd.read_csv(test_path)
    predictions_path = fit_models(train_path, test_path, tmp_path / "artifacts", 42)
    predictions = pd.read_csv(predictions_path)
    model = joblib.load(tmp_path / "artifacts" / "models" / "b_model.joblib")
    reloaded_predictions = model.predict(
        test.loc[:, ["gender", "season", "usage", "articleType", "year"]]
    )

    assert predictions_path.is_file()
    assert (tmp_path / "artifacts" / "cv_predictions.csv").is_file()
    assert set(predictions["row_id"]) == set(test["row_id"])
    assert len(predictions) == len(test)
    assert "one-off-holdout-category" in set(test["usage"])
    assert test["year"].isna().any()
    assert len(reloaded_predictions) == len(test)
    assert (tmp_path / "artifacts" / "models" / "a_equivalent.joblib").is_file()
    assert (tmp_path / "artifacts" / "models" / "majority_baseline.joblib").is_file()
