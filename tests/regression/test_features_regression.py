"""Regression tests for the fixture's deterministic split contract."""

from pathlib import Path

import pandas as pd
import pytest

from fashion_classifier.data import load_and_clean
from fashion_classifier.features import make_split


@pytest.mark.regression
def test_fixture_split_ids_and_rare_category_holdout(tmp_path: Path) -> None:
    """Pin seed-42 IDs, coverage, rare classes, and unseen test category."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "styles_small.csv"
    outputs = load_and_clean(fixture, tmp_path)
    train_path, test_path = make_split(outputs.clean_path, tmp_path, 0.2, 42)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    assert train["row_id"].tolist() == [
        12,
        33,
        9,
        0,
        4,
        16,
        17,
        5,
        13,
        11,
        1,
        2,
        30,
        3,
        29,
        23,
        32,
        22,
        18,
        25,
        6,
        20,
        34,
        7,
        10,
        14,
        28,
    ]
    assert test["row_id"].tolist() == [15, 19, 27, 26, 8, 24, 21]
    assert set(train["row_id"]).isdisjoint(test["row_id"])
    assert set(train["row_id"]) | set(test["row_id"]) == set(range(35)) - {31}
    assert train["baseColour"].value_counts().min() < 5
    assert "one-off-holdout-category" in set(test["usage"])
    assert "one-off-holdout-category" not in set(train["usage"])
