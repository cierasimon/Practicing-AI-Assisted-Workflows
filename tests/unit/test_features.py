"""Unit tests for feature selection and splitting."""

from pathlib import Path

import pandas as pd
import pytest

from fashion_classifier.features import OUTPUT_COLUMNS, make_split


@pytest.mark.unit
def test_split_excludes_missing_targets_and_only_keeps_allowlist(
    tmp_path: Path,
) -> None:
    """Exclude unlabeled rows and omit IDs and product-name text."""
    clean_path = tmp_path / "clean.csv"
    pd.DataFrame(
        {
            "row_id": range(8),
            "id": range(100, 108),
            "baseColour": [
                "Black",
                "Blue",
                None,
                "Red",
                "Black",
                "Blue",
                "Red",
                "Black",
            ],
            "gender": ["Men"] * 8,
            "productDisplayName": ["name"] * 8,
            "season": ["Fall"] * 8,
        }
    ).to_csv(clean_path, index=False)

    train_path, test_path = make_split(clean_path, tmp_path, 0.25, 42)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    assert list(train.columns) == list(OUTPUT_COLUMNS)
    assert list(test.columns) == list(OUTPUT_COLUMNS)
    assert 2 not in set(train["row_id"]) | set(test["row_id"])
    assert len(train) + len(test) == 7
    assert "id" not in train.columns
    assert "productDisplayName" not in train.columns
