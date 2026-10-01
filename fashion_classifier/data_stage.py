"""Entry point for the data preparation stage."""

import sys

from fashion_classifier.config import load_config
from fashion_classifier.data import load_and_clean
from fashion_classifier.paths import ensure_output_dirs, resolve_path


def main() -> None:
    """Load, audit, and profile the configured input dataset."""
    config = load_config()
    processed_dir, _ = ensure_output_dirs(config)
    try:
        outputs = load_and_clean(resolve_path(config.input_path), processed_dir)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error

    print(
        f"Accepted rows: {outputs.accepted_rows}; "
        f"rejected rows: {outputs.rejected_rows}"
    )
    print(f"Missing baseColour: {outputs.missing_targets}")
    print(f"Profile: {outputs.profile_path}")


if __name__ == "__main__":
    main()
