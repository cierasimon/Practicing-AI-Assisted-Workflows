"""Evaluation metrics, diagnostics, and controlled comparison reports."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

TEST_PREDICTION_COLUMNS = (
    "row_id",
    "true_label",
    "majority_prediction",
    "a_equivalent_prediction",
    "b_prediction",
)
CV_PREDICTION_COLUMNS = (
    "row_id",
    "fold",
    "true_label",
    "a_equivalent_prediction",
    "b_prediction",
)
MODEL_PREDICTIONS = {
    "majority": "majority_prediction",
    "a_equivalent": "a_equivalent_prediction",
    "b": "b_prediction",
}


def calculate_metrics(
    true_labels: list[str],
    predictions: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Calculate fixed-label classification metrics with safe zero division."""
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predictions)),
    }


def compute_cv_fold_metrics(cv_predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate paired fold macro F1 using labels present in each fold."""
    rows: list[dict[str, float | int]] = []
    for fold, fold_rows in cv_predictions.groupby("fold", sort=True):
        true_labels = fold_rows["true_label"].astype(str).tolist()
        labels = sorted(set(true_labels))
        a_score = float(
            f1_score(
                true_labels,
                fold_rows["a_equivalent_prediction"].astype(str).tolist(),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        b_score = float(
            f1_score(
                true_labels,
                fold_rows["b_prediction"].astype(str).tolist(),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        rows.append(
            {
                "fold": int(fold),
                "validation_rows": len(fold_rows),
                "a_equivalent_macro_f1": a_score,
                "b_macro_f1": b_score,
                "b_minus_a_macro_f1": b_score - a_score,
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions_path: Path,
    cv_predictions_path: Path,
    artifact_dir: Path,
) -> dict[str, object]:
    """Write fixed-test and OOF reports and return their computed metrics."""
    predictions = pd.read_csv(predictions_path)
    cv_predictions = pd.read_csv(cv_predictions_path)
    _require_columns(predictions, TEST_PREDICTION_COLUMNS, predictions_path)
    _require_columns(cv_predictions, CV_PREDICTION_COLUMNS, cv_predictions_path)
    if predictions["row_id"].duplicated().any():
        raise ValueError("Test predictions contain duplicate row_id values")
    if cv_predictions["row_id"].duplicated().any():
        raise ValueError("CV predictions must contain each training row exactly once")
    overlap = set(predictions["row_id"]) & set(cv_predictions["row_id"])
    if overlap:
        raise ValueError("Test rows appear in CV predictions")
    if set(cv_predictions["fold"]) != {1, 2, 3, 4, 5}:
        raise ValueError("CV predictions must contain all five folds")

    all_labels = sorted(
        set(cv_predictions["true_label"].astype(str))
        | set(predictions["true_label"].astype(str))
        | {
            str(label)
            for column in MODEL_PREDICTIONS.values()
            for label in predictions[column].dropna().unique()
        }
    )
    test_metrics, test_report, matrix_rows = _test_reports(predictions, all_labels)
    fold_metrics = compute_cv_fold_metrics(cv_predictions)
    cv_summary, cv_class_report = _cv_reports(cv_predictions, fold_metrics)
    decision = _comparison_decision(
        cv_summary["mean_b_minus_a_macro_f1"],
        test_metrics["b"]["macro_f1"] - test_metrics["a_equivalent"]["macro_f1"],
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "labels": all_labels,
        "test_metrics": test_metrics,
        "cv": cv_summary,
        "decision": decision,
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    fold_metrics.to_csv(artifact_dir / "cv_fold_metrics.csv", index=False)
    cv_class_report.to_csv(artifact_dir / "cv_classification_report.csv", index=False)
    test_report.to_csv(artifact_dir / "classification_report.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(artifact_dir / "confusion_matrix.csv", index=False)
    _write_comparison(artifact_dir / "comparison.md", metrics)
    return metrics


def add_data_quality_context(artifact_dir: Path, profile_path: Path) -> None:
    """Add malformed-row and missing-target counts to final evaluation reports."""
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    quality = {
        "accepted_rows": int(profile["accepted_rows"]),
        "rejected_rows": int(profile["rejected_rows"]),
        "missing_baseColour": int(profile["missing_baseColour"]),
    }
    metrics_path = artifact_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["data_quality"] = quality
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (artifact_dir / "comparison.md").open("a", encoding="utf-8") as report:
        report.write(
            "\n## Data quality\n\n"
            f"- Accepted CSV rows: {quality['accepted_rows']}\n"
            f"- Rejected malformed rows: {quality['rejected_rows']}\n"
            f"- Rows with missing `baseColour` excluded from supervised data: "
            f"{quality['missing_baseColour']}\n"
        )


def _test_reports(
    predictions: pd.DataFrame, labels: list[str]
) -> tuple[dict[str, dict[str, float]], pd.DataFrame, list[dict[str, object]]]:
    """Build fixed-test summary, per-class, and confusion-matrix rows."""
    true_labels = predictions["true_label"].astype(str).tolist()
    summary: dict[str, dict[str, float]] = {}
    report_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []

    for model_name, prediction_column in MODEL_PREDICTIONS.items():
        predicted = predictions[prediction_column].astype(str).tolist()
        summary[model_name] = calculate_metrics(true_labels, predicted, labels)
        report = classification_report(
            true_labels,
            predicted,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        for label in labels:
            label_report = report[label]
            report_rows.append(
                {
                    "model": model_name,
                    "class_label": label,
                    "precision": float(label_report["precision"]),
                    "recall": float(label_report["recall"]),
                    "f1": float(label_report["f1-score"]),
                    "support": int(label_report["support"]),
                }
            )
        matrix = confusion_matrix(true_labels, predicted, labels=labels)
        for true_index, true_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                matrix_rows.append(
                    {
                        "model": model_name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    return summary, pd.DataFrame(report_rows), matrix_rows


def _cv_reports(
    cv_predictions: pd.DataFrame, fold_metrics: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize paired folds and pooled class-level OOF diagnostics."""
    deltas = fold_metrics["b_minus_a_macro_f1"]
    summary: dict[str, Any] = {
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "mean_a_equivalent_macro_f1": float(
            fold_metrics["a_equivalent_macro_f1"].mean()
        ),
        "sample_std_a_equivalent_macro_f1": float(
            fold_metrics["a_equivalent_macro_f1"].std(ddof=1)
        ),
        "mean_b_macro_f1": float(fold_metrics["b_macro_f1"].mean()),
        "sample_std_b_macro_f1": float(fold_metrics["b_macro_f1"].std(ddof=1)),
        "mean_b_minus_a_macro_f1": float(deltas.mean()),
        "sample_std_b_minus_a_macro_f1": float(deltas.std(ddof=1)),
    }
    labels = sorted(cv_predictions["true_label"].astype(str).unique())
    rows: list[dict[str, object]] = []
    for model_name, prediction_column in (
        ("a_equivalent", "a_equivalent_prediction"),
        ("b", "b_prediction"),
    ):
        predicted = cv_predictions[prediction_column].astype(str).tolist()
        true_labels = cv_predictions["true_label"].astype(str).tolist()
        report = classification_report(
            true_labels,
            predicted,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        summary[f"pooled_oof_{model_name}_macro_f1"] = float(
            f1_score(
                true_labels,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        for label in labels:
            class_report = report[label]
            support = int(class_report["support"])
            rows.append(
                {
                    "model": model_name,
                    "class_label": label,
                    "precision": float(class_report["precision"]),
                    "recall": float(class_report["recall"]),
                    "f1": float(class_report["f1-score"]),
                    "support": support,
                    "low_support": support < 5,
                }
            )
    return summary, pd.DataFrame(rows)


def _comparison_decision(cv_delta: float, test_delta: float) -> str:
    """Classify improvement only when CV and fixed-test deltas agree."""
    if cv_delta > 0 and test_delta > 0:
        return "improvement supported by positive CV and fixed-test macro-F1 deltas"
    if (cv_delta > 0) != (test_delta > 0):
        return "inconclusive: CV and fixed-test macro-F1 deltas disagree"
    return "no improvement observed: both macro-F1 deltas are non-positive"


def _write_comparison(path: Path, metrics: dict[str, object]) -> None:
    """Write a concise controlled-comparison report from computed metrics."""
    test_metrics = metrics["test_metrics"]
    cv = metrics["cv"]
    folds = cv["fold_metrics"]
    test_delta = (
        test_metrics["b"]["macro_f1"] - test_metrics["a_equivalent"]["macro_f1"]
    )
    lines = [
        "# Fashion color model comparison",
        "",
        (
            "Historical Repo A accuracy: approximately 0.23, qualified context only. "
            "Repo A treated missing target colors as `Unknown`; this controlled rerun "
            "excludes missing `baseColour` rows."
        ),
        "",
        "## Fixed test split",
        "",
        "| Candidate | Accuracy | Macro F1 | Weighted F1 | Balanced accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name in ("majority", "a_equivalent", "b"):
        score = test_metrics[model_name]
        lines.append(
            f"| {model_name} | {score['accuracy']:.4f} | "
            f"{score['macro_f1']:.4f} | {score['weighted_f1']:.4f} | "
            f"{score['balanced_accuracy']:.4f} |"
        )
    a_test = test_metrics["a_equivalent"]
    b_test = test_metrics["b"]
    lines.extend(
        [
            "",
            (
                "On this fixed test split, B's accuracy was "
                f"{_metric_direction(b_test['accuracy'], a_test['accuracy'])} "
                "than A-equivalent's, while B's macro F1 was "
                f"{_metric_direction(b_test['macro_f1'], a_test['macro_f1'])} "
                "and its balanced accuracy was "
                f"{_metric_direction(b_test['balanced_accuracy'], a_test['balanced_accuracy'])}."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Paired five-fold CV",
            "",
            "Each fold macro F1 uses only labels present in that fold's validation truth.",
            "",
            "| Fold | A-equivalent macro F1 | B macro F1 | B minus A | Validation rows |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in folds:
        lines.append(
            f"| {fold['fold']} | {fold['a_equivalent_macro_f1']:.4f} | "
            f"{fold['b_macro_f1']:.4f} | {fold['b_minus_a_macro_f1']:.4f} | "
            f"{fold['validation_rows']} |"
        )
    lines.extend(
        [
            "",
            (
                f"CV mean A-equivalent macro F1: {cv['mean_a_equivalent_macro_f1']:.4f} "
                f"(sample SD {cv['sample_std_a_equivalent_macro_f1']:.4f})."
            ),
            (
                f"CV mean B macro F1: {cv['mean_b_macro_f1']:.4f} "
                f"(sample SD {cv['sample_std_b_macro_f1']:.4f})."
            ),
            f"Mean paired CV B-minus-A delta: {cv['mean_b_minus_a_macro_f1']:.4f}.",
            f"Fixed-test B-minus-A macro-F1 delta: {test_delta:.4f}.",
            "",
            f"**Conclusion:** {metrics['decision']}.",
            "",
            (
                "The fixed test split was evaluated once and was not used for tuning, "
                "feature selection, or model selection."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_direction(value: float, reference: float) -> str:
    """Describe a metric relative to its controlled-comparison reference."""
    if value > reference:
        return "higher"
    if value < reference:
        return "lower"
    return "equal"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    """Raise a readable error when an artifact lacks required columns."""
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Artifact {path} is missing columns: {sorted(missing)}")
