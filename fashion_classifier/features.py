"""Feature selection and deterministic outer train/test split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PREDICTOR_COLUMNS = ("gender", "season", "usage", "articleType", "year")
OUTPUT_COLUMNS = ("row_id", "baseColour", *PREDICTOR_COLUMNS)


def make_split(
    clean_path: Path,
    processed_dir: Path,
    test_size: float,
    seed: int,
) -> tuple[Path, Path]:
    """Write eligible rows to a deterministic train/test split."""
    if not clean_path.is_file():
        raise FileNotFoundError(f"Clean data file not found: {clean_path}")

    clean = pd.read_csv(clean_path)
    if "baseColour" not in clean.columns:
        raise ValueError("Required column 'baseColour' is missing from clean data")
    if "row_id" not in clean.columns:
        raise ValueError("Required column 'row_id' is missing from clean data")

    for column in PREDICTOR_COLUMNS:
        if column not in clean.columns:
            clean[column] = pd.NA

    missing_target_count = int(clean["baseColour"].isna().sum())
    eligible = clean.loc[clean["baseColour"].notna(), list(OUTPUT_COLUMNS)].copy()
    train, test = train_test_split(
        eligible,
        test_size=test_size,
        random_state=seed,
        stratify=None,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Excluded rows with missing baseColour: {missing_target_count}")
    print(f"Eligible rows: {len(eligible)}")
    return train_path, test_path
