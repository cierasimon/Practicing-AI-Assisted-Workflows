"""Configuration and path unit tests."""

from pathlib import Path

import pytest

from fashion_classifier.config import Config, load_config
from fashion_classifier.paths import ensure_output_dirs, resolve_path


@pytest.mark.unit
def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep documented defaults stable."""
    for name in (
        "FASHION_INPUT",
        "FASHION_PROCESSED_DIR",
        "FASHION_ARTIFACTS_DIR",
        "FASHION_SEED",
        "FASHION_TEST_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert load_config() == Config(
        Path("styles.csv"), Path("data/processed"), Path("artifacts"), 42, 0.2
    )


@pytest.mark.unit
def test_load_config_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read paths and split settings from the environment."""
    monkeypatch.setenv("FASHION_INPUT", "inputs/small.csv")
    monkeypatch.setenv("FASHION_PROCESSED_DIR", "tmp/processed")
    monkeypatch.setenv("FASHION_ARTIFACTS_DIR", "tmp/artifacts")
    monkeypatch.setenv("FASHION_SEED", "7")
    monkeypatch.setenv("FASHION_TEST_SIZE", "0.25")

    config = load_config()

    assert config.input_path == Path("inputs/small.csv")
    assert config.processed_dir == Path("tmp/processed")
    assert config.artifacts_dir == Path("tmp/artifacts")
    assert config.seed == 7
    assert config.test_size == 0.25


@pytest.mark.unit
def test_resolve_relative_path(tmp_path: Path) -> None:
    """Resolve relative paths from an explicit base directory."""
    assert (
        resolve_path(Path("data/input.csv"), tmp_path)
        == (tmp_path / "data/input.csv").resolve()
    )


@pytest.mark.unit
def test_ensure_output_dirs_creates_configured_directories(tmp_path: Path) -> None:
    """Create configured output directories and return their paths."""
    config = Config(
        Path("styles.csv"),
        tmp_path / "processed",
        tmp_path / "artifacts",
        42,
        0.2,
    )

    processed_dir, artifacts_dir = ensure_output_dirs(config)

    assert processed_dir.is_dir()
    assert artifacts_dir.is_dir()


@pytest.mark.unit
def test_invalid_test_size_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject split sizes that cannot define a holdout."""
    monkeypatch.setenv("FASHION_TEST_SIZE", "1")

    with pytest.raises(ValueError, match="FASHION_TEST_SIZE"):
        load_config()
