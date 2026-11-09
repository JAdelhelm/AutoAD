"""Marks for missing values, computed before any imputation."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from autoad._validation import as_frame, check_n_features, remember_input

_MODES = ("count", "columns")


class MissingValueMarker(TransformerMixin, BaseEstimator):
    """Mark missing values so that they reach the detector instead of vanishing in imputation.

    ``mode="count"`` returns one column ``missing_count`` with the number of missing
    values per row. ``mode="columns"`` returns one 0/1 column ``<name>_MISSING`` per
    input column.

    Unlike ``sklearn.impute.MissingIndicator(features="missing-only")``, every column is
    watched: a value that is missing during ``transform`` is reported even if its column
    had no missing values during ``fit``.

    Parameters
    ----------
    mode : {"count", "columns"}, default="count"
    """

    def __init__(self, mode="count"):
        self.mode = mode

    def fit(self, X, y=None):
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}.")
        remember_input(self, X)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        frame = as_frame(X)
        check_n_features(self, frame)
        mask = frame.isna().to_numpy()
        if self.mode == "count":
            return mask.sum(axis=1, keepdims=True).astype(float)
        return mask.astype(float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if self.mode == "count":
            return np.asarray(["missing_count"], dtype=object)
        names = _check_feature_names_in(self, input_features)
        return np.asarray([f"{name}_MISSING" for name in names], dtype=object)
