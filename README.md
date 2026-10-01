# Fashion Product Color Classifier

This repository turns the Fashion Product CSV into an auditable, staged classifier workflow. Each stage reads and writes explicit files, and all models share the same eligible rows and persisted 80/20 split.

## Setup

Requirements: Python 3.10 or newer, pip, and GNU Make on `PATH`. On Windows, install GNU Make separately (for example through MSYS2, Chocolatey, or Scoop), then open a PowerShell terminal where `make` is available. GNU Make is the public interface on Windows; Make recipes invoke Python modules and do not use POSIX cleanup or copy commands.

From the repository root:

```powershell
make --version
make install
make test
make lint
```

To use the Python launcher on Windows, override the executable as needed, for example `make test PYTHON="py -3"`.

## Configuration

All paths are resolved from the current working directory unless absolute paths are provided.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `FASHION_INPUT` | `styles.csv` | Input CSV path |
| `FASHION_PROCESSED_DIR` | `data/processed` | Clean and split CSV/profile outputs |
| `FASHION_ARTIFACTS_DIR` | `artifacts` | Models, predictions, and reports |
| `FASHION_SEED` | `42` | Deterministic split and CV seed |
| `FASHION_TEST_SIZE` | `0.2` | Unstratified outer holdout fraction |

PowerShell example:

```powershell
$env:FASHION_INPUT = 'styles.csv'
$env:FASHION_PROCESSED_DIR = 'data/processed'
$env:FASHION_ARTIFACTS_DIR = 'artifacts'
$env:FASHION_SEED = '42'
$env:FASHION_TEST_SIZE = '0.2'
```

## Pipeline

Run individual stages in order, or run the complete workflow with `make run`.

| Target | Module | Main outputs |
| --- | --- | --- |
| `make data` | `fashion_classifier.data_stage` | `clean.csv`, `rejected_rows.csv`, `profile.json` |
| `make features` | `fashion_classifier.features_stage` | `train.csv`, `test.csv` |
| `make train` | `fashion_classifier.train_stage` | `models/*.joblib`, `cv_predictions.csv`, `predictions.csv` |
| `make evaluate` | `fashion_classifier.evaluate_stage` | Metrics JSON, fold/class reports, confusion matrix, comparison Markdown |
| `make run` | Prerequisites: data, features, train, evaluate | All outputs above |
| `make clean` | `fashion_classifier.clean_stage` | Removes configured generated outputs and Python/test caches |

The data stage assigns stable accepted-row IDs, audits over-wide malformed records, adds absent optional predictors as null columns, and reports missing values. The Python CSV parser exposes malformed fields to the audit callback, but malformed quoting that prevents parsing may still fail the stage rather than be recoverable. Rows without `baseColour` remain visible in the clean data/profile but are excluded from supervised splitting. The feature stage uses only `gender`, `season`, `usage`, `articleType`, and `year`; row IDs and `productDisplayName` are never model features.

Training compares a Repo A-equivalent random forest using gender and season, a metadata model using the full allowlist with `balanced_subsample` class weights, and a most-frequent baseline. Imputation and one-hot encoding are fitted inside each training fold. Five-fold stratified out-of-fold predictions use only the outer training partition; the fixed test split is predicted once. Unseen categories are ignored safely. Evaluation reports accuracy, macro and weighted F1, balanced accuracy, per-class support/metrics, pooled OOF metrics for rare classes, paired fold deltas, and the fixed-test comparison. No plots or benchmark outputs are produced.

## Make Targets

`make install` installs dependencies from `requirements.txt`. `make test` runs unit, regression, and integration tests against the small fixture; tests do not read the full `styles.csv`. `make lint` runs Ruff. `make format` applies Ruff formatting. `make data`, `make features`, `make train`, and `make evaluate` run their corresponding Python modules. `make run` executes those targets in dependency order. `make clean` removes the configured processed/artifact output directories and `__pycache__`/`.pytest_cache` directories while preserving the raw input and test fixtures.

## Model Limitations

The target is `baseColour`, and rows without that label are not modeled. The historical Repo A score of approximately 0.23 treated missing targets as an `Unknown` class, so it is context rather than a directly matched result. The controlled A-equivalent rerun uses the same eligible rows and split as B.

Product names are intentionally excluded to prevent target-color leakage. Rare colors remain in the data and may have weak or zero recall; five-fold stratification is best-effort when a class has fewer than five training examples. The fixed holdout is not used for tuning. Improvement is supported only when both the mean paired CV macro-F1 delta and the fixed-test macro-F1 delta are positive; disagreement is inconclusive.

On the measured fixed-test comparison, B's accuracy is lower than the A-equivalent model's, while B's macro F1 and balanced accuracy are higher. This is a metric tradeoff, not an across-the-board improvement claim; the improvement decision rule above remains unchanged.

## Manual Smoke Tests

These are human-run workflows against the repository's root `styles.csv`; they are separate from the fixture-based test suite. Run them from PowerShell in the repository root.

### How to run

#### Stage 1: Setup

```powershell
make --version
make install
make lint
```

#### Stage 2: Data

```powershell
$env:FASHION_INPUT = 'styles.csv'
make data
```

#### Stage 3: Split

```powershell
make features
```

#### Stage 4: Train

```powershell
make train
```

#### Stage 5: Evaluate

```powershell
make evaluate
```

#### Stage 6: End to end and cleanup

```powershell
$env:FASHION_INPUT = 'styles.csv'
make run
make clean
```

Inspect `data/processed/` and `artifacts/` before cleanup.

## Evidence and Handoff TODOs

- **Manual smoke test results:** TODO, to be completed by the human Tester after running the commands above.
- **Measured Repo A vs Repo B results:** TODO, copy actual metrics with dataset, seed, and split details after the human-run smoke test. Do not infer results from the fixture.
- **Architect, Builder, and Tester contributions:** TODO, describe each role's actual contribution.
- **AI recommendation accepted:** TODO, record one recommendation and why it was accepted.
- **AI recommendation rejected or changed:** TODO, record one recommendation and the decision made.
- **Independent verification:** TODO, describe how the result was checked independently.

## Manual smoke test

Run on Windows (Git Bash, GNU Make) from a clean state with the root `styles.csv`:

    make test lint run

Result: PASS. 31 tests passed (4 expected rare-class stratification warnings, left visible by design); Ruff reported no issues; all four stages ran.

| Check | Observed |
| --- | --- |
| Rows accepted / rejected | 44,424 / 22 |
| Missing `baseColour` (excluded) | 15 |
| Eligible rows | 44,409 |
| Train / test (80/20, seed 42) | 35,527 / 8,882 |
| Rarest colour in the training split | 3 members (flagged, not merged or dropped) |

## Results: Repo A vs Repo B

Note: B does not improve raw accuracy; it improves macro F1 and balanced accuracy, and improvement is claimed only when both the mean paired CV macro-F1 delta and the fixed-test macro-F1 delta are positive.

Same eligible rows and the same fixed test split for all three models. Repo A's historical ~0.23 accuracy used a different missing-label treatment and is context only.

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced accuracy |
| --- | --- | --- | --- | --- |
| Majority baseline | 0.2212 | 0.0079 | 0.0802 | 0.0217 |
| A-equivalent (gender + season) | 0.2307 | 0.0145 | 0.0986 | 0.0279 |
| B (metadata + class weights) | 0.0976 | 0.0743 | 0.1039 | 0.2320 |

Five-fold CV on the training split (macro F1, A-equivalent vs B): fold deltas 0.0670, 0.0537, 0.0684, 0.0701, 0.0558; mean 0.0630, sample SD 0.0076. B is ahead in all five folds. Fixed-test delta: +0.0598. Both are positive, so the pre-set rule supports an improvement claim.

Interpretation: B is better at predicting less common colours (macro F1 and balanced accuracy up), but its overall accuracy is lower than both the A-equivalent and the majority baseline, so this is a trade-off rather than an across-the-board gain. Absolute scores remain low: gender, season, usage, article type and year carry limited signal about colour, and product names were excluded on purpose to avoid colour-word leakage.