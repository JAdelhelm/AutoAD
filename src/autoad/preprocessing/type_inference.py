"""Learn one type per column on the training data and cast later inputs the same way."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

NUMERIC = "numeric"
BOOL = "bool"
DATETIME = "datetime"
CATEGORICAL = "categorical"

#: Strings that stand for a missing value (compared after stripping whitespace).
MISSING_STRINGS = frozenset({"", "nan", "NaN", "NAN", "None", "null", "NULL", "<NA>", "N/A", "n/a"})

_DIGIT = re.compile(r"\d")


def to_datetime(series: pd.Series) -> pd.Series:
    """Parse ``series`` as datetimes. Values that cannot be parsed become NaT."""
    if is_datetime64_any_dtype(series.dtype):
        return series
    with warnings.catch_warnings():
        # pandas warns when it cannot infer one format and parses element by element.
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            return pd.to_datetime(series, errors="coerce")
        except (TypeError, ValueError):
            # For example values with different time zones.
            return pd.to_datetime(series, errors="coerce", utc=True)


def clean_missing(series: pd.Series) -> pd.Series:
    """Replace placeholder strings such as ``""`` or ``"nan"`` by NaN in text columns."""
    if not (
        is_object_dtype(series.dtype) or is_string_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return series
    values = series.to_numpy(dtype=object, copy=True)
    placeholder = np.fromiter(
        (isinstance(v, str) and v.strip() in MISSING_STRINGS for v in values),
        dtype=bool,
        count=len(values),
    )
    values[placeholder | pd.isna(values)] = np.nan
    return pd.Series(values, index=series.index, name=series.name, dtype=object)


def _bool_value(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return 1.0
        if text == "false":
            return 0.0
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return float(value)
    return np.nan


def _is_bool_like(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    return isinstance(value, str) and value.strip().lower() in ("true", "false")


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _as_bool(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=object)
    result = np.array([np.nan if pd.isna(v) else _bool_value(v) for v in values], dtype=float)
    return pd.Series(result, index=series.index, name=series.name)


def _as_categorical(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=object, copy=True)
    present = ~pd.isna(values)
    values[present] = [str(v) for v in values[present]]
    values[~present] = np.nan
    return pd.Series(values, index=series.index, name=series.name, dtype=object)


_CASTS = {
    NUMERIC: _as_numeric,
    BOOL: _as_bool,
    DATETIME: to_datetime,
    CATEGORICAL: _as_categorical,
}


class TypeInferenceTransformer(TransformerMixin, BaseEstimator):
    """Learn a type for every column during ``fit`` and cast later inputs the same way.

    The types are ``"numeric"``, ``"bool"``, ``"datetime"`` and ``"categorical"``.
    They are decided in this order:

    1. Columns in ``datetime_columns`` are datetime, columns in ``numerical_columns``
       are numeric.
    2. Columns with a bool, numeric or datetime dtype keep that type.
    3. For text, object and category columns the non-missing values decide:
       all booleans (or ``"true"``/``"false"``) give bool, all numbers (also numbers
       stored as text, such as ``"2001"``) give numeric, and datetime is chosen only
       if at least ``datetime_threshold`` of the values contain a digit and parse as
       a date. Everything else is categorical.

    Placeholder strings for missing values (``""``, ``"nan"``, ``"None"``, ...) count
    as missing. During ``transform`` every column is cast to the type learned in
    ``fit``; values that do not fit that type become missing.

    Parameters
    ----------
    datetime_columns : list of str, optional
    numerical_columns : list of str, optional
    datetime_threshold : float, default=0.95

    Attributes
    ----------
    column_types_ : dict
        Column name to type.
    """

    def __init__(self, datetime_columns=None, numerical_columns=None, datetime_threshold=0.95):
        self.datetime_columns = datetime_columns
        self.numerical_columns = numerical_columns
        self.datetime_threshold = datetime_threshold

    def fit(self, X, y=None):
        frame = self._check_frame(X)
        datetime_columns = list(self.datetime_columns or [])
        numerical_columns = list(self.numerical_columns or [])
        unknown = [c for c in datetime_columns + numerical_columns if c not in frame.columns]
        if unknown:
            raise ValueError(f"Columns not found in X: {unknown}")
        both = [c for c in datetime_columns if c in numerical_columns]
        if both:
            raise ValueError(f"Columns cannot be both datetime and numerical: {both}")
        self.column_types_ = {
            col: self._infer(col, clean_missing(frame[col]), datetime_columns, numerical_columns)
            for col in frame.columns
        }
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "column_types_")
        frame = self._check_frame(X)
        missing = [c for c in self.feature_names_in_ if c not in frame.columns]
        if missing:
            raise ValueError(f"X is missing columns seen during fit: {missing}")
        columns = {col: _CASTS[kind](clean_missing(frame[col])) for col, kind in self.column_types_.items()}
        return pd.DataFrame(columns, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "column_types_")
        return np.asarray(self.feature_names_in_, dtype=object)

    @staticmethod
    def _check_frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TypeInferenceTransformer expects a pandas DataFrame.")
        return X

    def _infer(self, col, series, datetime_columns, numerical_columns) -> str:
        if col in datetime_columns:
            return DATETIME
        if col in numerical_columns:
            return NUMERIC
        if is_bool_dtype(series.dtype):
            return BOOL
        if is_numeric_dtype(series.dtype):
            return NUMERIC
        if is_datetime64_any_dtype(series.dtype):
            return DATETIME
        values = series.dropna()
        if values.empty:
            return NUMERIC
        if all(_is_bool_like(v) for v in values):
            return BOOL
        if pd.to_numeric(values, errors="coerce").notna().all():
            return NUMERIC
        if self._looks_like_datetime(values):
            return DATETIME
        return CATEGORICAL

    def _looks_like_datetime(self, values: pd.Series) -> bool:
        # Words such as "today" or "March" parse as dates, real dates contain digits.
        has_digit = np.fromiter((bool(_DIGIT.search(str(v))) for v in values), dtype=bool, count=len(values))
        if has_digit.mean() < self.datetime_threshold:
            return False
        parsed = to_datetime(values.astype(object))
        return bool(parsed.notna().mean() >= self.datetime_threshold)
