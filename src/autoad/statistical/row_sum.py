"""Sum of all columns per row."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from autoad._validation import check_n_features, remember_input, to_float_array


class RowSum(TransformerMixin, BaseEstimator):
    """Sum all columns of a row into one column named ``name``.

    Used to turn per-column outlier flags into the number of flagged values per row.
    Missing values count as 0.
    """

    def __init__(self, name="Total"):
        self.name = name

    def fit(self, X, y=None):
        remember_input(self, X)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = to_float_array(X)
        check_n_features(self, values)
        return np.nansum(values, axis=1, keepdims=True)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        return np.asarray([self.name], dtype=object)
