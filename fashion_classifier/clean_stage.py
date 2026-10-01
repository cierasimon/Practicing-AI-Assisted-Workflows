"""Cross-platform cleanup for configured generated pipeline outputs."""

import os
import shutil
import sys
from pathlib import Path

from fashion_classifier.config import load_config
from fashion_classifier.paths import resolve_path


def clean_generated_outputs(
    processed_dir: Path,
    artifacts_dir: Path,
    project_root: Path,
    protected_paths: tuple[Path, ...],
) -> list[Path]:
    """Remove generated output directories without touching protected inputs."""
    root = project_root.resolve()
    removed: list[Path] = []
    output_paths = {processed_dir.resolve(), artifacts_dir.resolve()}
    for output_path in sorted(output_paths, key=str):
        if output_path == root or output_path in root.parents:
            raise ValueError(
                f"Refusing to remove project root or its parent: {output_path}"
            )
        for protected_path in protected_paths:
            protected = protected_path.resolve()
            if (
                output_path == protected
                or protected in output_path.parents
                or output_path in protected.parents
            ):
                raise ValueError(f"Refusing to remove protected path: {output_path}")
        if output_path.exists():
            if not output_path.is_dir():
                raise ValueError(f"Configured output is not a directory: {output_path}")
            shutil.rmtree(output_path)
            removed.append(output_path)
    return removed


def remove_python_caches(project_root: Path) -> list[Path]:
    """Remove Python bytecode and pytest cache directories below the project."""
    removed: list[Path] = []
    for current_root, directory_names, _ in os.walk(project_root):
        directory_names[:] = [name for name in directory_names if name != ".git"]
        for cache_name in ("__pycache__", ".pytest_cache"):
            if cache_name in directory_names:
                cache_path = Path(current_root) / cache_name
                shutil.rmtree(cache_path)
                directory_names.remove(cache_name)
                removed.append(cache_path)
    return removed


def main() -> None:
    """Remove configured outputs and generated Python/test caches."""
    config = load_config()
    root = Path.cwd().resolve()
    try:
        removed_outputs = clean_generated_outputs(
            resolve_path(config.processed_dir),
            resolve_path(config.artifacts_dir),
            root,
            (resolve_path(config.input_path), root / "tests" / "fixtures"),
        )
        removed_caches = remove_python_caches(root)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error

    for path in removed_outputs:
        print(f"Removed generated output: {path}")
    print(f"Removed cache directories: {len(removed_caches)}")


if __name__ == "__main__":
    main()
