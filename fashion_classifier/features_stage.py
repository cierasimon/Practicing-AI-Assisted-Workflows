"""Entry point for deterministic feature selection and splitting."""

import sys

import pandas as pd

from fashion_classifier.config import load_config
from fashion_classifier.features import make_split
from fashion_classifier.paths import resolve_path


def main() -> None:
    """Persist the configured eligible-row train/test split."""
    config = load_config()
    processed_dir = resolve_path(config.processed_dir)
    try:
        train_path, test_path = make_split(
            processed_dir / "clean.csv",
            processed_dir,
            config.test_size,
            config.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train rows: {len(train)}; test rows: {len(test)}; seed: {config.seed}")


if __name__ == "__main__":
    main()
