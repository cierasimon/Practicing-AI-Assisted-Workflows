"""Entry point for evaluation and comparison report generation."""

import sys

from fashion_classifier.config import load_config
from fashion_classifier.evaluation import add_data_quality_context, evaluate_predictions
from fashion_classifier.paths import resolve_path


def main() -> None:
    """Evaluate OOF and fixed-test predictions and print measured metrics."""
    config = load_config()
    artifact_dir = resolve_path(config.artifacts_dir)
    processed_dir = resolve_path(config.processed_dir)
    try:
        metrics = evaluate_predictions(
            artifact_dir / "predictions.csv",
            artifact_dir / "cv_predictions.csv",
            artifact_dir,
        )
        add_data_quality_context(artifact_dir, processed_dir / "profile.json")
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error

    for fold in metrics["cv"]["fold_metrics"]:
        print(
            f"Fold {fold['fold']}: A-equivalent macro F1="
            f"{fold['a_equivalent_macro_f1']:.4f}, B macro F1="
            f"{fold['b_macro_f1']:.4f}, delta={fold['b_minus_a_macro_f1']:.4f}"
        )
    print(
        f"CV mean delta: {metrics['cv']['mean_b_minus_a_macro_f1']:.4f}; "
        f"sample SD: {metrics['cv']['sample_std_b_minus_a_macro_f1']:.4f}"
    )
    for model_name, scores in metrics["test_metrics"].items():
        print(
            f"{model_name}: accuracy={scores['accuracy']:.4f}, "
            f"macro F1={scores['macro_f1']:.4f}, "
            f"weighted F1={scores['weighted_f1']:.4f}, "
            f"balanced accuracy={scores['balanced_accuracy']:.4f}"
        )
    print(f"Conclusion: {metrics['decision']}")


if __name__ == "__main__":
    main()
