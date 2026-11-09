"""Frequency encoding for categorical columns with many distinct values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from autoad._validation import as_frame, check_n_features, remember_input

_MISSING = "__autoad_missing__"


def _keys(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=object, copy=True)
    values[pd.isna(values)] = _MISSING
    return pd.Series(values, index=series.index, dtype=object)


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace every category by its relative frequency in the training data.

    Meant for columns with many distinct values, where a binary or one-hot code would
    add many dimensions without meaning. Categories that were not seen during ``fit``
    get 0.0, so rare and new values end up at the low end. Missing values count as a
    category of their own.

    Attributes
    ----------
    frequencies_ : list of dict
        Relative frequency of every category, per input column.
    """

    def fit(self, X, y=None):
        frame = as_frame(X)
        remember_input(self, X)
        self.frequencies_ = [
            _keys(frame.iloc[:, i]).value_counts(normalize=True).to_dict()
            for i in range(frame.shape[1])
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, "frequencies_")
        frame = as_frame(X)
        check_n_features(self, frame)
        result = np.empty(frame.shape, dtype=float)
        for i, frequencies in enumerate(self.frequencies_):
            mapped = _keys(frame.iloc[:, i]).map(frequencies)
            result[:, i] = mapped.to_numpy(dtype=float, na_value=0.0)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "frequencies_")
        names = _check_feature_names_in(self, input_features)
        return np.asarray([f"{name}_FREQ" for name in names], dtype=object)
