"""Environment-backed pipeline configuration."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Paths and deterministic split settings for pipeline stages."""

    input_path: Path
    processed_dir: Path
    artifacts_dir: Path
    seed: int
    test_size: float


def load_config() -> Config:
    """Load validated configuration from environment variables."""
    seed = int(os.environ.get("FASHION_SEED", "42"))
    test_size = float(os.environ.get("FASHION_TEST_SIZE", "0.2"))
    if not 0 < test_size < 1:
        raise ValueError("FASHION_TEST_SIZE must be between 0 and 1")

    return Config(
        input_path=Path(os.environ.get("FASHION_INPUT", "styles.csv")),
        processed_dir=Path(os.environ.get("FASHION_PROCESSED_DIR", "data/processed")),
        artifacts_dir=Path(os.environ.get("FASHION_ARTIFACTS_DIR", "artifacts")),
        seed=seed,
        test_size=test_size,
    )
