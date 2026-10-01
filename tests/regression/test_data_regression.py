"""Regression tests for stable cleaning and audit outputs."""

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion_classifier.data import load_and_clean


@pytest.mark.regression
def test_fixture_counts_and_null_accounting(tmp_path: Path) -> None:
    """Keep the fixture's malformed-row and missingness cases visible."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "styles_small.csv"
    outputs = load_and_clean(fixture, tmp_path)
    clean = pd.read_csv(outputs.clean_path)
    rejected = pd.read_csv(outputs.rejected_path)
    profile = json.loads(outputs.profile_path.read_text(encoding="utf-8"))

    assert outputs.accepted_rows == 35
    assert outputs.rejected_rows == 1
    assert outputs.missing_targets == 1
    assert clean["row_id"].tolist() == list(range(35))
    assert "overflow-field" in rejected.loc[0, "source_text"]
    assert "Expected 10 fields, received 11" in rejected.loc[0, "reason"]
    assert profile["missing_counts"]["baseColour"] == 1
    assert profile["missing_counts"]["year"] == 2
    assert profile["missing_counts"]["usage"] == 1
    assert profile["missing_counts"]["productDisplayName"] == 2
