# Changelog

## 0.2.0 - 2026-11-10

A rewrite that fixes wrong results and turns AutoAD into a scikit-learn estimator.
The API is not compatible with 1.0.0.

### Breaking changes

- Configuration moved from `fit()` to the constructor:
  `AutoAD(detector=IForest(), datetime_columns=[...], ...)`. `clf_ad` is now
  `detector`, `remove_columns_no_variance` is now `drop_constant`, `pipeline_type` was
  removed.
- `fit()` returns the estimator instead of the internal pipeline.
- `transform()` always returns the feature matrix. Scores come from
  `decision_function()`, labels from `predict()`, and the annotated table from the new
  `report()`, which keeps the row order instead of sorting by `AD_score`.
- `feature_importances` raises `AttributeError` instead of printing a message.
- Package layout `src/autoad`; import with `from autoad import AutoAD`. Modules are
  snake_case, `pipelines/dummy` is now `preprocessing`, `MedianAbsolutDeviation` is now
  `MedianAbsoluteDeviation`.
- Version 1.0.0 was reset to 0.x until the API is stable.

### Fixed

- Tukey and MAD computed one median/quartile over all columns together; they now work
  per column. Before, every row in the example had `Tukey_Total = 1`.
- No scaling before the detector; the feature matrix is now scaled with `RobustScaler`
  (configurable). Before, LOF gave all normal rows of the example the same score.
- MAD = 0 flagged every value that differs from the median; it now falls back to the
  mean absolute deviation. Tukey with IQR = 0 falls back to wider quantiles.
- A constant column broke `transform()` ("Column names must be identical!"). Constant
  columns are now found with `nunique() <= 1` instead of `std <= 0.1`.
- `transform()` ran no type inference; types are now learned in `fit()` and applied in
  `transform()`. Column mismatches are reported before transforming, with the missing
  and unexpected columns.
- `transform()` modified the caller's DataFrame.
- `pipeline_type="categorical"`/`"timeseries"` with a detector raised `KeyError`.
- `fit_transform()` skipped type inference, column removal and the detector.
- Numbers stored as text (`"2001"`) were parsed as dates.
- Columns of type `category` crashed, `string` columns were dropped silently and `bool`
  columns bypassed the missing-value marker.
- Missing values that only occur in new data no longer crash `transform()`
  (`MissingIndicator(features="missing-only")`) or vanish silently: `report()` lists them
  in `Missing_Columns`.
- Features that are constant in the training data (for example `missing_count` without
  missing values, or the BinaryEncoder's bit for unknown categories) are removed before
  the detector. They made PyOD's PCA return infinite scores on 17 of the 19 datasets of
  a one-off evaluation (not part of this repository).
- Importing AutoAD no longer disables all warnings, changes the global scikit-learn
  configuration or prints to stdout.
- pandas 3 support (`fillna(method=)`, `infer_datetime_format`, new string dtype,
  microsecond datetimes).

### Added

- `decision_function()`, `predict()`, `report()` and `get_feature_names_out()`.
- `report()` lists the columns that triggered Tukey and MAD per row, the columns with
  missing values (`Missing_Columns`) and categories not seen during `fit`
  (`Unseen_Columns`).
- `dropped_features_` lists the features removed because they are constant in training.
- `findings(X)` returns the findings behind `report()` per cell, as boolean DataFrames.
- `duplicates(X, tolerance=0.0)` finds rows that duplicate another row: KNN with 2
  neighbours on the feature matrix, where a low score (distance to the closest other row)
  marks a duplicate. Rows are grouped; ID-like and excluded columns are not compared.
- Web interface in `webui/` (Django): upload a file, run a detector, explore the
  anomalies and duplicates in charts and tables and check new files with the same model.
- ID-like columns are dropped (`drop_id_columns`).
- Frequency encoding for categorical columns with many distinct values.
- Date features only include hour, minute and second when a column contains a time of day.
- Tests for every fixed bug plus a quality test with injected anomalies.
- `pyproject.toml` (distribution name `autoad-framework`) with minimum versions instead
  of a pinned `requirements.txt`; extras `dev` and `examples`.
- GitHub Actions: ruff and pytest on Linux and Windows, Python 3.11 to 3.14, a run with
  the declared minimum versions of the dependencies, and a run of the example notebook.
- The example notebook uses synthetic data and needs no Kaggle account.

### Removed

- Unused `ZTransformerMean*`, `SpearmanCheck` and `NaNColumnCreator*` (and with them the
  undeclared `bitstring` dependency).
- `TimeSeriesImputer` (forward fill); date parts are imputed with the median.
- `requirements.txt` with the Windows-only `pywin32` pin.
