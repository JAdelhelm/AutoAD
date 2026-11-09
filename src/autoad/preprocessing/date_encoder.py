"""Numeric date and time components from datetime columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from autoad._validation import as_frame, check_n_features, remember_input
from autoad.preprocessing.type_inference import to_datetime

DATE_UNITS = ("YEAR", "MONTH", "DAY", "WKDAY")
TIME_UNITS = ("HOUR", "MINUTE", "SECOND")
_ATTRIBUTES = {
    "YEAR": "year",
    "MONTH": "month",
    "DAY": "day",
    "WKDAY": "day_of_week",
    "HOUR": "hour",
    "MINUTE": "minute",
    "SECOND": "second",
}


class DateEncoder(TransformerMixin, BaseEstimator):
    """Turn every datetime column into numeric components.

    Every column yields ``<name>_YEAR``, ``_MONTH``, ``_DAY`` and ``_WKDAY``
    (0 = Monday). ``_HOUR``, ``_MINUTE`` and ``_SECOND`` are added only for columns
    that contain a time of day during ``fit`` (``include_time="auto"``), so pure
    dates do not produce constant features. A time of day that first appears after
    ``fit`` is therefore not encoded. Missing or unparsable values become NaN.

    Parameters
    ----------
    include_time : {"auto", True, False}, default="auto"

    Attributes
    ----------
    units_ : list of tuple
        Encoded units per input column.
    """

    def __init__(self, include_time="auto"):
        self.include_time = include_time

    def fit(self, X, y=None):
        if self.include_time not in ("auto", True, False):
            raise ValueError("include_time must be 'auto', True or False.")
        frame = as_frame(X)
        remember_input(self, X)
        self.units_ = []
        for i in range(frame.shape[1]):
            if self.include_time == "auto":
                parsed = to_datetime(frame.iloc[:, i]).dropna().dt
                has_time = bool(((parsed.hour != 0) | (parsed.minute != 0) | (parsed.second != 0)).any())
            else:
                has_time = bool(self.include_time)
            self.units_.append(DATE_UNITS + TIME_UNITS if has_time else DATE_UNITS)
        return self

    def transform(self, X):
        check_is_fitted(self, "units_")
        frame = as_frame(X)
        check_n_features(self, frame)
        names = _check_feature_names_in(self, None)
        columns = {}
        for i, (name, units) in enumerate(zip(names, self.units_, strict=True)):
            parsed = to_datetime(frame.iloc[:, i]).dt
            for unit in units:
                component = getattr(parsed, _ATTRIBUTES[unit])
                columns[f"{name}_{unit}"] = component.to_numpy(dtype=float, na_value=np.nan)
        return pd.DataFrame(columns, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "units_")
        names = _check_feature_names_in(self, input_features)
        return np.asarray(
            [f"{name}_{unit}" for name, units in zip(names, self.units_, strict=True) for unit in units],
            dtype=object,
        )
