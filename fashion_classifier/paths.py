"""Path resolution and output-directory setup."""

from pathlib import Path

from fashion_classifier.config import Config


def resolve_path(path: Path, base_dir: Path | None = None) -> Path:
    """Resolve a configured path against the working or supplied directory."""
    base = Path.cwd() if base_dir is None else base_dir
    return path if path.is_absolute() else (base / path).resolve()


def ensure_output_dirs(config: Config) -> tuple[Path, Path]:
    """Create and return configured processed and artifact directories."""
    processed_dir = resolve_path(config.processed_dir)
    artifacts_dir = resolve_path(config.artifacts_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, artifacts_dir
