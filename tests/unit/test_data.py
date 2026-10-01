"""Unit tests for data validation and profiling."""

import pandas as pd
import pytest

from fashion_classifier.data import profile_data


@pytest.mark.unit
def test_profile_counts_missing_values_and_targets() -> None:
    """Count null predictors and target labels without imputing them."""
    frame = pd.DataFrame(
        {"baseColour": ["Black", "Blue", None], "season": ["Fall", None, "Spring"]}
    )

    profile = profile_data(frame)

    assert profile["rows"] == 3
    assert profile["missing_counts"] == {"baseColour": 1, "season": 1}
    assert profile["target_counts"] == {"Black": 1, "Blue": 1, "<NULL>": 1}


@pytest.mark.unit
def test_profile_requires_base_colour() -> None:
    """Reject frames without the supervised target column."""
    with pytest.raises(ValueError, match="baseColour"):
        profile_data(pd.DataFrame({"season": ["Fall"]}))
