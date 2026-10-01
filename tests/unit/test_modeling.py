"""Unit tests for configured model pipelines and fold constraints."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from fashion_classifier.modeling import build_pipeline, cross_validate_models


@pytest.mark.unit
def test_pipeline_uses_imputation_unknown_safe_encoding_and_fixed_forest() -> None:
    """Keep estimator settings and preprocessing behavior fixed."""
    pipeline = build_pipeline(["gender", "season", "year"])
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    categorical = next(
        transformer
        for name, transformer, _ in preprocessor.transformers
        if name == "categorical"
    )

    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.get_params()["n_estimators"] == 100
    assert classifier.get_params()["max_depth"] == 10
    assert classifier.get_params()["random_state"] == 42
    assert classifier.get_params()["class_weight"] is None
    assert isinstance(categorical.named_steps["encoder"], OneHotEncoder)
    assert categorical.named_steps["encoder"].handle_unknown == "ignore"
    assert (
        build_pipeline(["gender"], "balanced_subsample")
        .named_steps["classifier"]
        .class_weight
        == "balanced_subsample"
    )


@pytest.mark.unit
def test_pipeline_fits_nulls_and_predicts_unseen_categories() -> None:
    """Impute training values and ignore categories first seen at prediction."""
    pipeline = build_pipeline(["gender", "year"])
    train = pd.DataFrame(
        {
            "gender": ["Men", None, "Women", "Men", "Women"],
            "year": [2011, None, 2012, 2013, 2014],
        }
    )
    target = ["Black", "Blue", "Black", "Blue", "Black"]

    pipeline.fit(train, target)
    predictions = pipeline.predict(
        pd.DataFrame({"gender": ["Unseen", None], "year": [None, 2015]})
    )

    assert len(predictions) == 2


@pytest.mark.unit
def test_cross_validation_rejects_fewer_than_five_rows(tmp_path: Path) -> None:
    """Fail clearly instead of silently reducing the required fold count."""
    train_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "row_id": range(4),
            "baseColour": ["Black", "Blue", "Black", "Blue"],
            "gender": ["Men"] * 4,
            "season": ["Fall"] * 4,
            "usage": ["Casual"] * 4,
            "articleType": ["Shirts"] * 4,
            "year": [2011] * 4,
        }
    ).to_csv(train_path, index=False)

    with pytest.raises(ValueError, match="at least five"):
        cross_validate_models(train_path, tmp_path / "oof.csv", 42)
