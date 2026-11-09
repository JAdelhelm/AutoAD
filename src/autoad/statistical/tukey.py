"""Tukey's fences, applied column by column."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from autoad._validation import check_n_features, remember_input, to_float_array


class TukeyTransformer(TransformerMixin, BaseEstimator):
    """Flag values outside Tukey's fences, separately for every column.

    A value is flagged with 1 when it lies below ``Q1 - factor * IQR`` or above
    ``Q3 + factor * IQR``. The quartiles are learned per column during ``fit``.
    Missing values are ignored and never flagged.

    When the IQR of a column is 0 (for example a binary column in which at least
    about 75 % of the values are equal), these fences would flag every other value.
    The fences are then built from ``fallback_quantiles`` instead. If that range is
    0 as well, the column is skipped and never flagged.

    Parameters
    ----------
    factor : float, default=1.5
        Multiple of the IQR that defines the fences.
    fallback_quantiles : tuple of (float, float) or None, default=(0.05, 0.95)
        Quantile pair used for columns whose IQR is 0. ``None`` skips such columns.

    Attributes
    ----------
    lower_, upper_ : ndarray of shape (n_features_in_,)
        Fences per column.
    skipped_ : ndarray of bool, shape (n_features_in_,)
        Columns that are never flagged because they have no spread.
    """

    def __init__(self, factor=1.5, fallback_quantiles=(0.05, 0.95)):
        self.factor = factor
        self.fallback_quantiles = fallback_quantiles

    def fit(self, X, y=None):
        values = to_float_array(X)
        remember_input(self, X)
        with warnings.catch_warnings():
            # All-NaN columns give NaN quantiles; they end up in skipped_.
            warnings.simplefilter("ignore", RuntimeWarning)
            low, high = np.nanquantile(values, [0.25, 0.75], axis=0)
            spread = high - low
            no_spread = ~(spread > 0)
            if self.fallback_quantiles is not None and no_spread.any():
                fallback_low, fallback_high = np.nanquantile(values, list(self.fallback_quantiles), axis=0)
                low = np.where(no_spread, fallback_low, low)
                high = np.where(no_spread, fallback_high, high)
                spread = high - low
        self.skipped_ = ~(spread > 0)
        self.lower_ = low - self.factor * spread
        self.upper_ = high + self.factor * spread
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        values = to_float_array(X)
        check_n_features(self, values)
        with np.errstate(invalid="ignore"):
            flags = (values < self.lower_) | (values > self.upper_)
        flags[:, self.skipped_] = False
        return flags.astype(np.int64)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "lower_")
        names = _check_feature_names_in(self, input_features)
        return np.asarray([f"{name}_TUKEY" for name in names], dtype=object)
