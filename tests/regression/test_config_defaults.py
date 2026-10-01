"""Regression checks for the pipeline's stable defaults."""

from pathlib import Path

import pytest

from fashion_classifier.config import load_config


@pytest.mark.regression
def test_root_input_and_split_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve root CSV, seed, and holdout defaults."""
    for name in (
        "FASHION_INPUT",
        "FASHION_SEED",
        "FASHION_TEST_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_config()

    assert config.input_path == Path("styles.csv")
    assert config.seed == 42
    assert config.test_size == 0.2
