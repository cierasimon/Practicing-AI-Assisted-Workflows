"""Entry point for cross-validation, model fitting, and test predictions."""

import sys

from fashion_classifier.config import load_config
from fashion_classifier.modeling import fit_models
from fashion_classifier.paths import resolve_path


def main() -> None:
    """Train both candidates and persist OOF and fixed-test predictions."""
    config = load_config()
    processed_dir = resolve_path(config.processed_dir)
    artifact_dir = resolve_path(config.artifacts_dir)
    try:
        fit_models(
            processed_dir / "train.csv",
            processed_dir / "test.csv",
            artifact_dir,
            config.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
