"""Leakage-resistant model pipelines, cross-validation, and predictions."""

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fashion_classifier.features import PREDICTOR_COLUMNS

A_FEATURE_COLUMNS = ("gender", "season")


def build_pipeline(
    feature_columns: Sequence[str], class_weight: str | None = None
) -> Pipeline:
    """Build training-fitted imputing, encoding, and random-forest steps."""
    categorical_columns = [column for column in feature_columns if column != "year"]
    numeric_columns = [column for column in feature_columns if column == "year"]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                ),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight=class_weight,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def cross_validate_models(train_path: Path, output_path: Path, seed: int) -> Path:
    """Write five-fold training-only OOF predictions for both forests."""
    train = pd.read_csv(train_path)
    _validate_split(train, train_path)
    if len(train) < 5:
        raise ValueError(
            "Five-fold cross-validation requires at least five outer-training rows"
        )

    target = train["baseColour"]
    all_features = train.loc[:, list(PREDICTOR_COLUMNS)]
    a_features = train.loc[:, list(A_FEATURE_COLUMNS)]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_rows: list[dict[str, object]] = []

    for fold_number, (fit_indices, validation_indices) in enumerate(
        splitter.split(all_features, target), start=1
    ):
        fold_target = target.iloc[fit_indices]
        a_model = build_pipeline(A_FEATURE_COLUMNS)
        b_model = build_pipeline(PREDICTOR_COLUMNS, class_weight="balanced_subsample")
        a_model.fit(a_features.iloc[fit_indices], fold_target)
        b_model.fit(all_features.iloc[fit_indices], fold_target)
        a_predictions = a_model.predict(a_features.iloc[validation_indices])
        b_predictions = b_model.predict(all_features.iloc[validation_indices])
        for index, a_prediction, b_prediction in zip(
            validation_indices, a_predictions, b_predictions, strict=True
        ):
            oof_rows.append(
                {
                    "row_id": train.iloc[index]["row_id"],
                    "fold": fold_number,
                    "true_label": target.iloc[index],
                    "a_equivalent_prediction": a_prediction,
                    "b_prediction": b_prediction,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(oof_rows).to_csv(output_path, index=False)
    return output_path


def fit_models(
    train_path: Path,
    test_path: Path,
    artifact_dir: Path,
    seed: int,
) -> Path:
    """Fit final candidates on outer training rows and predict fixed test rows."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    _validate_split(train, train_path)
    _validate_split(test, test_path)
    if len(train) < 5:
        raise ValueError(
            "Five-fold cross-validation requires at least five outer-training rows"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    cross_validate_models(train_path, artifact_dir / "cv_predictions.csv", seed)

    target = train["baseColour"]
    a_model = build_pipeline(A_FEATURE_COLUMNS)
    b_model = build_pipeline(PREDICTOR_COLUMNS, class_weight="balanced_subsample")
    a_model.fit(train.loc[:, list(A_FEATURE_COLUMNS)], target)
    b_model.fit(train.loc[:, list(PREDICTOR_COLUMNS)], target)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train), 1)), target)
    baseline_predictions = baseline.predict(np.zeros((len(test), 1)))
    a_predictions = a_model.predict(test.loc[:, list(A_FEATURE_COLUMNS)])
    b_predictions = b_model.predict(test.loc[:, list(PREDICTOR_COLUMNS)])

    joblib.dump(a_model, model_dir / "a_equivalent.joblib")
    joblib.dump(b_model, model_dir / "b_model.joblib")
    joblib.dump(baseline, model_dir / "majority_baseline.joblib")
    predictions_path = artifact_dir / "predictions.csv"
    pd.DataFrame(
        {
            "row_id": test["row_id"],
            "true_label": test["baseColour"],
            "majority_prediction": baseline_predictions,
            "a_equivalent_prediction": a_predictions,
            "b_prediction": b_predictions,
        }
    ).to_csv(predictions_path, index=False)
    print(
        f"Trained A-equivalent, B, and majority baseline on {len(train)} rows; "
        f"predicted {len(test)} test rows"
    )
    return predictions_path


def _validate_split(frame: pd.DataFrame, path: Path) -> None:
    """Require row IDs, labels, and the complete feature allowlist."""
    required = {"row_id", "baseColour", *PREDICTOR_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Split file {path} is missing columns: {sorted(missing)}")
    if frame["baseColour"].isna().any():
        raise ValueError(f"Split file {path} contains missing baseColour labels")
