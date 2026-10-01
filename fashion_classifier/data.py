"""CSV loading, data-quality accounting, and profiling."""

import csv
import json
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd

OPTIONAL_COLUMNS = (
    "gender",
    "season",
    "usage",
    "articleType",
    "year",
    "productDisplayName",
)


@dataclass(frozen=True)
class DataOutputs:
    """Paths and row counts produced by the data stage."""

    clean_path: Path
    rejected_path: Path
    profile_path: Path
    accepted_rows: int
    rejected_rows: int
    missing_targets: int


def profile_data(frame: pd.DataFrame) -> dict[str, object]:
    """Return schema, missing-value, and target-count summaries."""
    if "baseColour" not in frame.columns:
        raise ValueError("Required column 'baseColour' is missing")

    target_counts = frame["baseColour"].value_counts(dropna=False)
    return {
        "rows": len(frame),
        "schema": [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in frame.dtypes.items()
        ],
        "missing_counts": {
            str(column): int(count) for column, count in frame.isna().sum().items()
        },
        "target_counts": {
            "<NULL>" if pd.isna(label) else str(label): int(count)
            for label, count in target_counts.items()
        },
    }


def load_and_clean(input_path: Path, processed_dir: Path) -> DataOutputs:
    """Read the CSV, audit over-wide rows, and write cleaned profile outputs."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rejected: list[dict[str, str]] = []
    expected_fields = len(pd.read_csv(input_path, nrows=0).columns)

    def capture_bad_line(fields: list[str]) -> None:
        source_buffer = StringIO(newline="")
        csv.writer(source_buffer, lineterminator="").writerow(fields)
        rejected.append(
            {
                "source_text": source_buffer.getvalue(),
                "reason": f"Expected {expected_fields} fields, received {len(fields)}",
            }
        )

    frame = pd.read_csv(
        input_path,
        engine="python",
        on_bad_lines=capture_bad_line,
    )
    if "baseColour" not in frame.columns:
        raise ValueError("Required column 'baseColour' is missing")

    missing_optional = [
        column for column in OPTIONAL_COLUMNS if column not in frame.columns
    ]
    if missing_optional:
        warnings.warn(
            "Optional predictor columns missing and added as null: "
            + ", ".join(missing_optional),
            stacklevel=2,
        )
        for column in missing_optional:
            frame[column] = pd.NA

    if "row_id" in frame.columns:
        raise ValueError("Input contains reserved column 'row_id'")
    frame.insert(0, "row_id", range(len(frame)))

    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_path = processed_dir / "clean.csv"
    rejected_path = processed_dir / "rejected_rows.csv"
    profile_path = processed_dir / "profile.json"
    frame.to_csv(clean_path, index=False)
    pd.DataFrame(rejected, columns=("source_text", "reason")).to_csv(
        rejected_path, index=False
    )

    profile = profile_data(frame)
    profile.update(
        {
            "accepted_rows": len(frame),
            "rejected_rows": len(rejected),
            "missing_baseColour": int(frame["baseColour"].isna().sum()),
        }
    )
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    return DataOutputs(
        clean_path=clean_path,
        rejected_path=rejected_path,
        profile_path=profile_path,
        accepted_rows=len(frame),
        rejected_rows=len(rejected),
        missing_targets=int(frame["baseColour"].isna().sum()),
    )
